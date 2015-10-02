#include "mainwindow.h"
#include <QApplication>
#include <GL/freeglut_std.h>

#include "meshvisualizer.h"
#include "multilinearreconstructor.hpp"

vector<int> LoadIndices(const string& filename) {
  ifstream fin(filename);
  vector<int> indices;
  istream_iterator<int> iter(fin);
  std::copy(iter, istream_iterator<int>(), back_inserter(indices));
  cout << indices.size() << " landmarks loaded." << endl;
  return indices;
}

namespace std {
istream& operator>>(istream& is, Constraint2D& c) {
  is >> c.data.x >> c.data.y;
  return is;
}
}

vector<Constraint2D> LoadConstraints(const string& filename) {
  ifstream fin(filename);
  int num_constraints;
  fin >> num_constraints;

  istream_iterator<Constraint2D> iter(fin);
  istream_iterator<Constraint2D> iter_end;
  vector<Constraint2D> constraints;
  std::copy(iter, iter_end, back_inserter(constraints));

  std::for_each(constraints.begin(), constraints.end(), [](Constraint2D& c) {
    c.vidx = -1;
    c.weight = 1.0;
    // The coordinates are one-based. Fix them.
    c.data.x -= 1.0;
    c.data.y -= 1.0;
  });

  cout << num_constraints << " constraints expected. "
       << constraints.size() << " constraints loaded." << endl;
  assert(num_constraints == constraints.size());
  return constraints;
}

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  glutInit(&argc, argv);
  SingleImageReconstructor<Constraint2D> recon;
  recon.LoadModel("/home/phg/Data/Multilinear/blendshape_core.tensor");
  recon.LoadPriors("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                   "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  auto landmarks = LoadIndices("/home/phg/Data/Multilinear/landmarks_73.txt");
  recon.SetIndices(landmarks);
  auto constraints = LoadConstraints("/home/phg/Data/InternetRecon/yaoming/4.pts");
  recon.SetConstraints(constraints);
  QImage img("/home/phg/Data/InternetRecon/yaoming/4.jpg");
  cout << "image size: " << img.width() << "x" << img.height() << endl;
  recon.SetImageSize(img.width(), img.height());

  BasicMesh mesh("/home/phg/Data/Multilinear/template.obj");
  MeshVisualizer w("reconstruction result", mesh);
  w.BindConstraints(constraints);
  w.BindImage(img);
  w.BindLandmarks(landmarks);
  w.resize(img.width(), img.height());
  w.show();

  return a.exec();
}
