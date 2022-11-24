cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(jm-test)

# set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/newlibtorch/libtorch)
set(CMAKE_PREFIX_PATH /home/kmsjames/very-big-storage/hojin/pytorch/perfisol/libtorch)
find_package(Torch REQUIRED)

set(COMMON_DIR /home/kmsjames/very-big-storage/jimin/pytorch/include)

file(GLOB SOURCES "src/*")
#set(SOURCES src/multithreadtest.cpp) #manual add

include_directories(${COMMON_DIR})

add_executable(jm-test ${SOURCES})
target_link_libraries(jm-test "${TORCH_LIBRARIES}")
set_property(TARGET jm-test PROPERTY CXX_STANDARD 14)
