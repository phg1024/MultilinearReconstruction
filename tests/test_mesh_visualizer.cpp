#include <QApplication>
#include <GL/freeglut_std.h>
#include "../meshvisualizer.h"

int main(int argc, char** argv) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);
  BasicMesh mesh("/home/phg/Data/Multilinear/template.obj");
  MeshVisualizer w("template mesh", mesh);
  w.show();
  return a.exec();
}
