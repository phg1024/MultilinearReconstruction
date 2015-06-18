#include "mainwindow.h"
#include <QApplication>

#include "multilinearmodelbuilder.h"
#include "multilinearreconstructor.hpp"

#include "test_all.h"

int main(int argc, char *argv[])
{
  /*
  testAll();
  */

  /*
  MultilinearModelBuilder builder;
  builder.build();
  return 0;
  */

  SingleImageReconstructor<Constraint2D> recon;
  recon.loadModel("/home/phg/Data/Multilinear/blendshape_core.tensor");
  recon.loadPriors("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                   "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");

  QApplication a(argc, argv);
  MainWindow w;
  w.show();

  return a.exec();
}
