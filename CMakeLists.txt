cmake_minimum_required(VERSION 3.18)

project(mpm2dcloth LANGUAGES CXX CUDA)

set(ZS_ENABLE_CUDA On)
set(ZS_ENABLE_OPENMP On)
set(ZS_ENABLE_OPENVDB Off)

add_subdirectory(zpc)

add_executable(mpmcloth2d)
target_sources(mpmcloth2d
    PRIVATE     cloth2d.cu
)
target_link_libraries(mpmcloth2d zensim)