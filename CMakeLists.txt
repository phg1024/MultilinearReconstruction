cmake_minimum_required(VERSION 2.8.11)

project(MultilinearReconstruction)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

set(CMAKE_AUTOUIC ON)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()

# Eigen
include_directories(/usr/include/eigen3)

# MKL
include_directories(${MKL_INCLUDE_DIRS})
set(MKLROOT "/home/phg/SDKs/intel/composer_xe_2015.2.164/mkl")
set(MKLLIBS "-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -ldl -lpthread -lm")

# PhGLib
include_directories("/home/phg/SDKs/PhGLib/include")
set(PhGLib "-L/home/phg/SDKs/PhGLib/lib -lPhGLib")

# Qt5
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)

# Tell CMake to create the helloworld executable
add_library(multilinearmodel multilinearmodel.cpp)
target_link_libraries(multilinearmodel
        Qt5::Widgets
        Qt5::OpenGL
        ${MKLLIBS}
        ${PhGLib})

add_library(basicmesh basicmesh.cpp)
target_link_libraries(basicmesh
        Qt5::Widgets
        Qt5::OpenGL
        ${MKLLIBS}
        ${PhGLib})

add_library(meshvisualizer meshvisualizer.cpp)
target_link_libraries(meshvisualizer
        Qt5::Widgets
        Qt5::OpenGL
        ${MKLLIBS}
        ${PhGLib})

add_library(tensor tensor.hpp)
target_link_libraries(tensor ${MKLLIBS})

add_executable(MultilinearModelBuilder multilinearmodelbuilder.cpp)
add_executable(MultilinearReconstruction main.cpp mainwindow.cpp multilinearreconstructor.hpp utils.hpp)

# Use the Widgets module from Qt 5.
target_link_libraries(MultilinearReconstruction multilinearmodel basicmesh tensor
                      Qt5::Core
                      Qt5::Widgets
                      Qt5::OpenGL
					  ${MKLLIBS}
					  ${PhGLib})

target_link_libraries(MultilinearModelBuilder multilinearmodel)

add_subdirectory(tests)