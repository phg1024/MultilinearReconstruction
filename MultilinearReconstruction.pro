#-------------------------------------------------
#
# Project created by QtCreator 2015-04-06T14:34:29
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MultilinearReconstruction
TEMPLATE = subdirs
SUBDIRS = tests

QMAKE_CXXFLAGS += -std=c++11

# PhGLib
INCLUDEPATH += /home/phg/SDKs/PhGLib/include
LIBS += -L/home/phg/SDKs/PhGLib/lib -lPhGLib

# ceres
INCLUDEPATH += /home/phg/SDKs/ceres-solver-1.10.0/include /usr/include/eigen3
LIBS += -L/home/phg/SDKs/ceres-solver-1.10.0/lib -lceres -lglog -gflags

# CHOLMOD
#INCLUDEPATH += /home/phg/SDKs/SuiteSparse/SuiteSparse_config /home/phg/SDKs/SuiteSparse/CHOLMOD/Include
#LIBS += -L/home/phg/SDKs/SuiteSparse/SuiteSparse_config -L/home/phg/SDKs/SuiteSparse/AMD/Lib -L/home/phg/SDKs/SuiteSparse/COLAMD/Lib -L/home/phg/SDKs/SuiteSparse/CHOLMOD/Lib -lcholmod -lamd -lcolamd -lsuitesparseconfig
LIBS += -lcholmod -lcamd -lamd -lccolamd -lcolamd -lsuitesparseconfig -lcxsparse

# CGAL
LIBS += -lCGAL -lboost_system

# GLEW
INCLUDEPATH += /home/phg/SDKs/glew-1.12.0/include
LIBS += -L/home/phg/SDKs/glew-1.12.0/lib -lGLEW

# GLUT
INCLUDEPATH += /home/phg/SDKs/freeglut-2.8.1/include
LIBS += -lGLU -lglut

# MKL
QMAKE_CXXFLAGS += -fopenmp -m64 -I${MKLROOT}/include
LIBS += -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -ldl -lpthread -lm

# eigen
INCLUDEPATH += /usr/include/eigen3

# glm
INCLUDEPATH += /home/phg/SDKs/glm-0.9.6.3

SOURCES += main.cpp\
        mainwindow.cpp \
    multilinearmodel.cpp \
    meshvisualizer.cpp \
    basicmesh.cpp

HEADERS  += mainwindow.h \
    tensor.hpp \
    blendshape_data.h \
    common.h \
    multilinearmodelbuilder.h \
    utils.hpp \
    test_tensor.h \
    test_all.h \
    multilinearmodel.h \
    multilinearreconstructor.hpp \
    constraints.h \
    costfunctions.h \
    meshvisualizer.h \
    basicmesh.h

FORMS    += mainwindow.ui
