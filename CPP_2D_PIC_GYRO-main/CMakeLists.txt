cmake_minimum_required(VERSION 3.16)

project(CPP_2D_PIC_GYRO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CPP_2D_PIC_GYRO_ENABLE_OPENMP "Enable OpenMP if available" ON)

file(GLOB_RECURSE CPP_2D_PIC_GYRO_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(cpp_2d_pic_gyro ${CPP_2D_PIC_GYRO_SOURCES})

target_include_directories(cpp_2d_pic_gyro PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(cpp_2d_pic_gyro PRIVATE -O3 -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(cpp_2d_pic_gyro PRIVATE /O2 /W4)
endif()

find_package(OpenMP QUIET)
if(CPP_2D_PIC_GYRO_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(cpp_2d_pic_gyro PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(cpp_2d_pic_gyro PRIVATE CPP_2D_PIC_GYRO_USE_OPENMP=1)
endif()

set_target_properties(cpp_2d_pic_gyro PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
