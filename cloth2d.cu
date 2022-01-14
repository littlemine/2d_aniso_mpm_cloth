// ref: https://zhuanlan.zhihu.com/p/414356129
#include "zensim/container/TileVector.hpp"
#include "zensim/container/Vector.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/Atomics.hpp"
#include "zensim/io/ParticleIO.hpp"

using T = float;
using vec2 = zs::vec<T, 2>;
using vec2i = zs::vec<int, 2>;
using mat2 = zs::vec<T, 2, 2>;

constexpr int dim = 2;
constexpr int ngrid = 256;
constexpr int nvoxels = 65536;
constexpr T dx = (T)0.00390625; // (T)1. / ngrid;
constexpr T dxinv = (T)256.;
constexpr T dt = (T)4e-5;

constexpr T rho = (T)1.;
constexpr T E = (T)5000.;      // stretch
constexpr T g_gamma = (T)500.; // shear
constexpr T k = (T)1000.;      // normal

// number of lines
constexpr auto N_Line = 12;
// line space distance
constexpr auto dlx = (T)0.009;
// type2 particle count per line
constexpr auto ln_type2 = 200;
// num segments/elements/particle ?
constexpr auto ln_type3 = 199; // ln_type2 - 1;

constexpr auto start_pos = vec2{(T)0.2, (T)0.602};

constexpr auto n_type2 = 2400; // N_Line * ln_type2;
constexpr auto n_type3 = 2388; // N_Line * ln_type3;

// line length
constexpr auto Length = (T)0.75;
// segment length
constexpr auto sl = (T)0.003768844; // Length / ln_type3;

constexpr T volume2 = 0.000007343; // dx * Length / (ln_type3 + ln_type2);
constexpr T volume3 = 0.000007343; // volume2;

constexpr auto n_segment = 2388; // n_type3;
constexpr mat2 rot90{0, -1, 1, 0};

constexpr vec2 circleCenter{(T)0.7, (T)0.2};
constexpr T circleRadius{(T)0.4};

// Gram–Schmidt Orthogonalization to the column vectors of a full column rank
// matrix yields the QR decomposition (it is decomposed into an orthogonal and a
// triangular matrix)
__forceinline__ __device__ auto QR2(mat2 mat) noexcept {
  auto c0 = vec2{mat(0, 0), mat(1, 0)};
  auto c1 = vec2{mat(0, 1), mat(1, 1)};
  T r11 = sqrt(c0.l2NormSqr() + (T)1e-6);
  auto q0 = c0 / r11;
  auto r12 = c1.dot(q0);
  auto q1 = c1 - r12 * q0;
  T r22 = sqrt(q1.l2NormSqr() + (T)1e-6);
  q1 = q1 / r22;
  auto Q = mat2{q0(0), q1(0), q0(1), q1(1)};
  auto R = mat2{r11, r12, 0, r22};
  return std::make_tuple(Q, R);
}

constexpr auto c2lo(int x, int y) noexcept { return x * ngrid + y; }
constexpr auto get_type2_from_type3(int i) noexcept {
  i += (i / ln_type3);
  return std::make_tuple(i, i + 1);
}

int main() {
  using namespace zs;
  (void)zs::Cuda::instance();

  constexpr auto memsrc = memsrc_e::device;
  constexpr auto space = execspace_e::cuda;
  constexpr auto execTag = wrapv<space>{};
  static_assert(is_backend_available(wrapv<space>{}),
                "execution space is not available");

  /// type 2
  using allocator_type = ZSPmrAllocator<false>; // virtual or not both fine
  Vector<vec2, allocator_type> x2{n_type2, memsrc, 0};
  Vector<vec2, allocator_type> v2{n_type2, memsrc, 0};
  Vector<mat2, allocator_type> C2{n_type2, memsrc, 0};

  /// type 3
  Vector<vec2, allocator_type> x3{n_type3, memsrc, 0};
  Vector<vec2, allocator_type> v3{n_type3, memsrc, 0};
  Vector<mat2, allocator_type> C3{n_type3, memsrc, 0};
  Vector<mat2, allocator_type> F3{n_type3, memsrc, 0};
  Vector<mat2, allocator_type> D3inv{n_type3, memsrc, 0};
  Vector<mat2, allocator_type> d3{n_type3, memsrc, 0};

  TileVector<T, 32, unsigned char, allocator_type> grid{
      {{"v", dim}, {"m", 1}, {"f", dim}}, nvoxels, memsrc, 0};

  auto cudaExec = par_exec(wrapv<space>{}).device(0).sync(true).profile(false);

  // init
  cudaExec(range(n_type2), [x = proxy<space>(x2), v = proxy<space>(v2),
                            C = proxy<space>(C2)] __device__(int pi) mutable {
    auto sq = pi / ln_type2;
    x[pi] =
        vec2{start_pos[0] + (pi - sq * ln_type2) * sl, start_pos[1] + sq * dlx};
    v[pi] = vec2::zeros();
    C[pi] = mat2::zeros();
  });
  cudaExec(range(n_segment),
           [x2 = proxy<space>(x2), x3 = proxy<space>(x3), v = proxy<space>(v3),
            F = proxy<space>(F3), C = proxy<space>(C3), d3 = proxy<space>(d3),
            D3inv = proxy<space>(D3inv)] __device__(int i) mutable {
             auto [l, r] = get_type2_from_type3(i);
             x3[i] = (T)0.5 * (x2[l] + x2[r]);
             v[i] = vec2::zeros();
             F[i] = mat2::identity();
             C[i] = mat2::zeros();

             auto dp0 = x2[r] - x2[l];
             constexpr mat2 rot90{0, -1, 1, 0};
             auto dp1 = rot90 * dp0;
             dp1 = dp1 / sqrt(dp1.l2NormSqr() + (T)1e-6);
             d3[i] = mat2{dp0[0], dp1[0], dp0[1], dp1[1]};
             D3inv[i] = inverse(d3[i]);
           });

  for (int it = 0; it != 50000; ++it) {
    // reset grid
    cudaExec(range(nvoxels),
             [grid = proxy<space>({}, grid)] __device__(int gi) mutable {
               grid("m", gi) = (T)0.;
               for (int d = 0; d != dim; ++d) {
                 grid("v", d, gi) = (T)0.;
                 grid("f", d, gi) = (T)0.;
               }
             });
    // p2g
    cudaExec(range(x2.size()),
             [vol = volume2, x = proxy<space>(x2), v = proxy<space>(v2),
              C = proxy<space>(C2),
              grid = proxy<space>({}, grid)] __device__(int pi) mutable {
               auto base = (x[pi] * dxinv - (T)0.5).cast<int>();
               auto fx = x[pi] * dxinv - base.cast<T>();
               vec2 w[3] = {(T)0.5 * ((T)1.5 - fx) * ((T)1.5 - fx),
                            (T)0.75 - (fx - (T)1.) * (fx - (T)1.),
                            (T)0.5 * (fx - (T)0.5) * (fx - (T)0.5)};
               auto affine = C[pi];
               auto mass = vol * rho;
               for (auto [i, j] : ndrange<dim>(3)) {
                 auto offset = vec2i{i, j};
                 auto gi = c2lo(base[0] + offset[0], base[1] + offset[1]);
                 auto weight = w[i][0] * w[j][1];
                 atomic_add(execTag, &grid("m", gi), weight * mass);
                 auto dpos = (offset.cast<T>() - fx) * dx;
                 auto mv_p = weight * mass * (v[pi] + affine * dpos);
                 for (int d = 0; d != dim; ++d)
                   atomic_add(execTag, &grid("v", d, gi), mv_p[d]);
               }
             });
    cudaExec(range(x3.size()),
             [vol = volume3, x = proxy<space>(x3), v = proxy<space>(v3),
              C = proxy<space>(C3),
              grid = proxy<space>({}, grid)] __device__(int pi) mutable {
               auto base = (x[pi] * dxinv - (T)0.5).cast<int>();
               auto fx = x[pi] * dxinv - base.cast<T>();
               vec2 w[3] = {(T)0.5 * ((T)1.5 - fx) * ((T)1.5 - fx),
                            (T)0.75 - (fx - (T)1.) * (fx - (T)1.),
                            (T)0.5 * (fx - (T)0.5) * (fx - (T)0.5)};
               auto affine = C[pi];
               auto mass = vol * rho;
               for (auto [i, j] : ndrange<dim>(3)) {
                 auto offset = vec2i{i, j};
                 auto gi = c2lo(base[0] + offset[0], base[1] + offset[1]);
                 auto weight = w[i][0] * w[j][1];
                 atomic_add(execTag, &grid("m", gi), weight * mass);
                 auto dpos = (offset.cast<T>() - fx) * dx;
                 auto mv_p = weight * mass * (v[pi] + affine * dpos);
                 for (int d = 0; d != dim; ++d)
                   atomic_add(execTag, &grid("v", d, gi), mv_p[d]);
               }
             });
    // force
    cudaExec(range(x3.size()),
             [volume2 = volume2, volume3 = volume3, x3 = proxy<space>(x3),
              x2 = proxy<space>(x2), v = proxy<space>(v3),
              F3 = proxy<space>(F3), C3 = proxy<space>(C3),
              grid = proxy<space>({}, grid), d3 = proxy<space>(d3),
              D3inv = proxy<space>(D3inv)] __device__(int pi) mutable {
               auto [l, r] = get_type2_from_type3(pi);
               auto base = (x3[pi] * dxinv - (T)0.5).cast<int>();
               auto fx = x3[pi] * dxinv - base.cast<T>();
               vec2 w[3] = {(T)0.5 * ((T)1.5 - fx) * ((T)1.5 - fx),
                            (T)0.75 - (fx - (T)1.) * (fx - (T)1.),
                            (T)0.5 * (fx - (T)0.5) * (fx - (T)0.5)};
               vec2 dw_dx_d[3] = {(fx - (T)1.5) * dxinv,
                                  (T)2. * ((T)1. - fx) * dxinv,
                                  (fx - (T)0.5) * dxinv};

               auto base_l = (x2[l] * dxinv - (T)0.5).cast<int>();
               auto fx_l = x2[l] * dxinv - base_l.cast<T>();
               vec2 w_l[3] = {(T)0.5 * ((T)1.5 - fx_l) * ((T)1.5 - fx_l),
                              (T)0.75 - (fx_l - (T)1.) * (fx_l - (T)1.),
                              (T)0.5 * (fx_l - (T)0.5) * (fx_l - (T)0.5)};

               auto base_r = (x2[r] * dxinv - (T)0.5).cast<int>();
               auto fx_r = x2[r] * dxinv - base_r.cast<T>();
               vec2 w_r[3] = {(T)0.5 * ((T)1.5 - fx_r) * ((T)1.5 - fx_r),
                              (T)0.75 - (fx_r - (T)1.) * (fx_r - (T)1.),
                              (T)0.5 * (fx_r - (T)0.5) * (fx_r - (T)0.5)};

               auto [Q, R] = QR2(F3[pi]);

               auto r11 = R(0, 0);
               auto r12 = R(0, 1);
               auto r22 = R(1, 1);

               /// critical!
               mat2 A{(E * r11 * (r11 - (T)1.) + g_gamma * r12 * r12),
                      g_gamma * r12 * r22, g_gamma * r12 * r22,
                      r22 <= (T)1. ? -k * ((T)1. - r22) * ((T)1. - r22) * r22
                                   : (T)0.};

               auto dphi_dF = Q * A * inverse(R).transpose();

               vec2 dp_c1{d3[pi](0, 1), d3[pi](1, 1)};

               vec2 dphi_dF_c1{dphi_dF(0, 1), dphi_dF(1, 1)};

               vec2 Dp_inv_c0{D3inv[pi](0, 0), D3inv[pi](0, 1)};

               for (auto [i, j] : ndrange<dim>(3)) {
                 auto offset = vec2i{i, j};
                 auto gi_l = c2lo(base_l[0] + offset[0], base_l[1] + offset[1]);
                 auto gi_r = c2lo(base_r[0] + offset[0], base_r[1] + offset[1]);
                 auto weight_l = w_l[i][0] * w_l[j][1];
                 auto weight_r = w_r[i][0] * w_r[j][1];
                 auto f_2 = dphi_dF * Dp_inv_c0;
                 for (int d = 0; d != dim; ++d) {
                   atomic_add(execTag, &grid("f", d, gi_l),
                              volume3 * weight_l * f_2[d]);
                   atomic_add(execTag, &grid("f", d, gi_r),
                              -volume3 * weight_r * f_2[d]);
                 }
                 auto gi = c2lo(base[0] + offset[0], base[1] + offset[1]);
                 vec2 dw_dx{dw_dx_d[i][0] * w[j][1], w[i][0] * dw_dx_d[j][1]};
                 // tech doc (15) part 2
                 auto v{-volume3 * dphi_dF_c1 * dot(dw_dx, dp_c1)};
                 for (int d = 0; d != dim; ++d)
                   atomic_add(execTag, &grid("f", d, gi), v[d]);
               }
             });
#if 1
    // force: bending
    cudaExec(range((ln_type2 - 2) * N_Line),
             [x2 = proxy<space>(x2),
              grid = proxy<space>({}, grid)] __device__(int p) mutable {
               auto nl = p / (ln_type2 - 2);
               auto v0 = p + nl * 2;
               auto v1 = v0 + 2;

               auto base_0 = (x2[v0] * dxinv - (T)0.5).cast<int>();
               auto fx_0 = x2[v0] * dxinv - base_0.cast<T>();
               vec2 w_0[3] = {(T)0.5 * ((T)1.5 - fx_0) * ((T)1.5 - fx_0),
                              (T)0.75 - (fx_0 - (T)1.) * (fx_0 - (T)1.),
                              (T)0.5 * (fx_0 - (T)0.5) * (fx_0 - (T)0.5)};

               auto base_1 = (x2[v1] * dxinv - (T)0.5).cast<int>();
               auto fx_1 = x2[v1] * dxinv - base_1.cast<T>();
               vec2 w_1[3] = {(T)0.5 * ((T)1.5 - fx_1) * ((T)1.5 - fx_1),
                              (T)0.75 - (fx_1 - (T)1.) * (fx_1 - (T)1.),
                              (T)0.5 * (fx_1 - (T)0.5) * (fx_1 - (T)0.5)};

               auto dir_x = x2[v1] - x2[v0];
               auto dist = sqrtf(dir_x.l2NormSqr() + (T)1e-9);
               dir_x = dir_x / dist;
               auto fn = dist - (T)2. * sl;
               auto f = (T)-1000. * fn * dir_x;

               for (auto [i, j] : ndrange<dim>(3)) {
                 auto offset = vec2i{i, j};
                 auto gi_0 = c2lo(base_0[0] + offset[0], base_0[1] + offset[1]);
                 auto gi_1 = c2lo(base_1[0] + offset[0], base_1[1] + offset[1]);
                 auto weight_0 = w_0[i][0] * w_0[j][1];
                 auto weight_1 = w_1[i][0] * w_1[j][1];
                 for (int d = 0; d != dim; ++d) {
                   atomic_add(execTag, &grid("f", d, gi_0), -weight_0 * f[d]);
                   atomic_add(execTag, &grid("f", d, gi_1), weight_1 * f[d]);
                 }
               }
             });
#endif

    // grid update
    cudaExec(
        range(nvoxels),
        [dt = dt, grid = proxy<space>({}, grid), circleCenter = circleCenter,
         circleRadius = circleRadius] __device__(int gi) mutable {
          constexpr int bound = 3;
          int i = gi / ngrid, j = gi % ngrid;
          if (grid("m", gi) > 0) {
            for (int d = 0; d != dim; ++d)
              grid("v", d, gi) =
                  (grid("v", d, gi) + grid("f", d, gi) * dt) / grid("m", gi);
            grid("v", 1, gi) -= (T)9.8 * dt;

            // circle collision
            auto dist = vec2{i * dx, j * dx} - circleCenter;
            if (auto l = dot(dist, dist); l < circleRadius * circleRadius) {
              dist = dist / sqrt(l);
              auto val =
                  zs::min((T)0., dot(grid.template pack<dim>("v", gi), dist));
              for (int d = 0; d != dim; ++d) {
                grid("v", d, gi) -= dist[d] * val;
                grid("v", d, gi) *= (T)0.9; // friction
              }
            }

            if (i < bound && grid("v", 0, gi) < 0)
              grid("v", 0, gi) = 0;
            if (i > ngrid - bound && grid("v", 0, gi) > 0)
              grid("v", 0, gi) = 0;
            if (j < bound && grid("v", 1, gi) < 0)
              grid("v", 1, gi) = 0;
            if (j > ngrid - bound && grid("v", 1, gi) > 0)
              grid("v", 1, gi) = 0;
          }
        });

    // g2p
    cudaExec(range(x2.size()),
             [dt = dt, x = proxy<space>(x2), v = proxy<space>(v2),
              C = proxy<space>(C2),
              grid = proxy<space>({}, grid)] __device__(int pi) mutable {
               auto base = (x[pi] * dxinv - (T)0.5).cast<int>();
               auto fx = x[pi] * dxinv - base.cast<T>();
               vec2 w[3] = {(T)0.5 * ((T)1.5 - fx) * ((T)1.5 - fx),
                            (T)0.75 - (fx - (T)1.) * (fx - (T)1.),
                            (T)0.5 * (fx - (T)0.5) * (fx - (T)0.5)};
               auto new_v = vec2::zeros();
               auto new_C = mat2::zeros();

               for (auto [i, j] : ndrange<dim>(3)) {
                 auto offset = vec2i{i, j};
                 auto dpos = (offset.cast<T>() - fx);
                 auto gi = c2lo(base[0] + offset[0], base[1] + offset[1]);
                 auto weight = w[i][0] * w[j][1];
                 auto gv = grid.template pack<dim>("v", gi);
                 new_v += weight * gv;
                 new_C += 4 * weight * dyadic_prod(gv, dpos) * dxinv;
               }
               v[pi] = new_v;
               x[pi] += new_v * dt;
               C[pi] = new_C;
             });

    cudaExec(range(x3.size()),
             [x = proxy<space>(x3), C = proxy<space>(C3),
              grid = proxy<space>({}, grid)] __device__(int pi) mutable {
               auto base = (x[pi] * dxinv - (T)0.5).cast<int>();
               auto fx = x[pi] * dxinv - base.cast<T>();
               vec2 w[3] = {(T)0.5 * ((T)1.5 - fx) * ((T)1.5 - fx),
                            (T)0.75 - (fx - (T)1.) * (fx - (T)1.),
                            (T)0.5 * (fx - (T)0.5) * (fx - (T)0.5)};
               auto new_C = mat2::zeros();

               for (auto [i, j] : ndrange<dim>(3)) {
                 auto offset = vec2i{i, j};
                 auto dpos = (offset.cast<T>() - fx);
                 auto gi = c2lo(base[0] + offset[0], base[1] + offset[1]);
                 auto weight = w[i][0] * w[j][1];
                 auto gv = grid.template pack<dim>("v", gi);
                 new_C += 4 * weight * dyadic_prod(gv, dpos) * dxinv;
               }
               C[pi] = new_C;
             });
    cudaExec(range(x3.size()),
             [x3 = proxy<space>(x3), v3 = proxy<space>(v3),
              x2 = proxy<space>(x2), v2 = proxy<space>(v2),
              F3 = proxy<space>(F3), C3 = proxy<space>(C3),
              grid = proxy<space>({}, grid), d3 = proxy<space>(d3),
              D3inv = proxy<space>(D3inv)] __device__(int pi) mutable {
               auto [l, r] = get_type2_from_type3(pi);
               v3[pi] = (T)0.5 * (v2[l] + v2[r]);
               x3[pi] = (T)0.5 * (x2[l] + x2[r]);

               auto dp1 = x2[r] - x2[l];
               vec2 dp2{d3[pi](0, 1), d3[pi](1, 1)};
               dp2 = dp2 + dt * (C3[pi] * dp2);
               d3[pi] = mat2{dp1[0], dp2[0], dp1[1], dp2[1]};
               F3[pi] = d3[pi] * D3inv[pi];
             });
    // return mapping
    cudaExec(range(x3.size()),
             [F3 = proxy<space>(F3), d3 = proxy<space>(d3),
              D3inv = proxy<space>(D3inv)] __device__(int pi) mutable {
               constexpr T cf = (T)0.05;
               auto [Q, R] = QR2(F3[pi]);
               auto r12 = R(0, 1);
               auto r22 = R(1, 1);

               if (r22 < 0) {
                 r12 = (T)0.;
                 r22 = zs::max(r22, -(T)1.);
               } else if (r22 > (T)1.) {
                 r12 = (T)0.;
                 r22 = (T)1.;
               } else {
                 auto rr = r12 * r12;
                 auto zz = cf * ((T)1. - r22) * ((T)1. - r22);
                 auto gamma_over_s = g_gamma / k;
                 auto f = gamma_over_s * gamma_over_s * rr - zz * zz;
                 if (f > 0) {
                   auto scale = zz / (gamma_over_s * sqrt(rr));
                   r12 *= scale;
                 }
               }

               R(0, 1) = r12;
               R(1, 1) = r22;
               F3[pi] = Q * R;
               d3[pi] = F3[pi] * inverse(D3inv[pi]);
             });

    // output
    if (it % 100 == 0) {
      std::string fn = fmt::format("step[{}]_cloths.bgeo", it / 100);
      fmt::print("writing {} with {} particles ({})\n", fn, x2.size(), n_type2);
      auto out = x2.clone(MemoryLocation{memsrc_e::host, -1});
      std::vector<std::array<T, dim>> ret(out.size());
      // sizeof(std::array<T, dim>) == sizeof(vec2)
      memcpy(ret.data(), out.data(), sizeof(vec2) * out.size());
      write_partio<T, dim>(fn, ret);
    }
  }

  return 0;
}