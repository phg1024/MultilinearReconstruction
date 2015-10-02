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

# Qt5
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)

# OpenGL
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
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

# Tell CMake to create the helloworld executable
add_executable(test_mesh_visualizer test_mesh_visualizer.cpp)
# Use the Widgets module from Qt 5.
target_link_libraries(test_mesh_visualizer basicmesh meshvisualizer
        Qt5::Core
        Qt5::Widgets
        Qt5::OpenGL
        ${MKLLIBS}
        ${PhGLib})

add_executable(test_tensors test_tensors.cpp)
target_link_libraries(test_tensors tensor)


link_directories(..)