cmake_minimum_required( VERSION 3.16 )
project(CMINUSF)

set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -std=c99")

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined -fsanitize=address")

set(default_build_type "Debug")

if(NOT(CMAKE_BUILD_TYPE_SHADOW STREQUAL CMAKE_BUILD_TYPE))
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}'")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    else()
        message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode")
    endif()

    set(CMAKE_BUILD_TYPE_SHADOW ${CMAKE_BUILD_TYPE} CACHE STRING "used to detect changes in build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DYYDEBUG=1")

find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)
# set(CMAKE_INSTALL_PREFIX "/data/home/zsj/MLIR-Related/llvm-project/build/default-install" CACHE PATH "Installation directory" FORCE)
set(LLVM_INSTALL_PREFIX "/data/home/zsj/MLIR-Related/llvm-project/build/cminusf-dev-install"
    CACHE PATH "LLVM/MLIR install prefix")
set(MLIR_DIR "${LLVM_INSTALL_PREFIX}/lib/cmake/mlir" CACHE PATH "Path to MLIR cmake files")
set(LLVM_DIR "${LLVM_INSTALL_PREFIX}/lib/cmake/llvm" CACHE PATH "Path to LLVM cmake files")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_CMAKE_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_CMAKE_DIR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "LLVM Include Dirs: ${LLVM_INCLUDE_DIRS}")
message(STATUS "MLIR Include Dirs: ${MLIR_INCLUDE_DIRS}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)
include(HandleLLVMOptions)

llvm_map_components_to_libnames(
    llvm_libs
    option
    support
    core
)
include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
INCLUDE_DIRECTORIES(
    include
    include/cminusfc
    include/common
    include/mlir
    ${CMAKE_CURRENT_BINARY_DIR}/include/
)

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(include)
add_subdirectory(src)
# add_subdirectory(tests)
