#-------------------------------------------------
#
# Project created by QtCreator 2015-04-06T14:34:29
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MultilinearReconstruction
TEMPLATE = app

QMAKE_CXXFLAGS += -std=c++11

# MKL
QMAKE_CXXFLAGS += -fopenmp -m64 -I${MKLROOT}/include
LIBS +=  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -liomp5 -ldl -lpthread -lm

SOURCES += main.cpp\
        mainwindow.cpp

INCLUDEPATH += /usr/include/eigen3

HEADERS  += mainwindow.h \
    tensor.hpp \
    blendshape_data.h \
    common.h \
    multilinearmodelbuilder.h \
    utils.hpp \
    test_tensor.h \
    test_all.h

FORMS    += mainwindow.ui
