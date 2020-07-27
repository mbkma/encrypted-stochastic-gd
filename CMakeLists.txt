cmake_minimum_required(VERSION 3.12)
project(sgd)

if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message(FATAL_ERROR "test requires at least g++-8")
endif()

find_package(ABY QUIET)
if(ABY_FOUND)
	message(STATUS "Found ABY")
elseif (NOT ABY_FOUND AND NOT TARGET ABY::aby)
	message("ABY was not found: add ABY subdirectory")
	add_subdirectory(extern/ABY)
endif()

find_package(eigen QUIET)
if(eigen_FOUND)
	message(STATUS "Found EIGEN")
elseif (NOT eigen_FOUND AND NOT TARGET eigen::eigen)
	message("EIGEN was not found: add EIGEN subdirectory")
	add_subdirectory(extern/eigen)
endif()

add_executable(main src/main.cpp src/sgd.cpp src/linear-model-generator.cpp)
target_link_libraries(main ABY::aby eigen)

target_compile_options(main PRIVATE "-Wall" "-Wextra" "-O0" "-g")

