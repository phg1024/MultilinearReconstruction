#include <QApplication>
#include <GL/freeglut_std.h>

#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"

#include "meshvisualizer.h"
#include "singleimagereconstructor.hpp"
#include "glog/logging.h"
#include "multiimagereconstructor.h"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  if( argc < 2 ) {
    cout << "Usage: ./MultiImageReconstruction setting_file" << endl;
    return -1;
  }

  MultiImageReconstructor<Constraint2D> recon;

  return a.exec();
}