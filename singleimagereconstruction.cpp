#include <QApplication>
#include <GL/freeglut_std.h>

#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"

#include "meshvisualizer.h"
#include "singleimagereconstructor.hpp"
#include "glog/logging.h"

vector<int> LoadIndices(const string& filename) {
  ifstream fin(filename);
  vector<int> indices;
  istream_iterator<int> iter(fin);
  std::copy(iter, istream_iterator<int>(), back_inserter(indices));
  for(int i=0;i<indices.size();++i) cout << indices[i] << endl;

  cout << indices.size() << " landmarks loaded." << endl;
  return indices;
}

vector<vector<int>> LoadContourIndices(const string& filename) {
  ifstream fin(filename);
  vector<string> lines;
  while( fin ) {
    string line;
    std::getline(fin, line);
    if( !line.empty() )
      lines.push_back(line);
  }

  vector<vector<int>> contour_indices(lines.size());
  std::transform(lines.begin(), lines.end(), contour_indices.begin(),
                 [](const string& line){
                   cout << "line: " << line << endl;
                   vector<string> parts;
                   boost::algorithm::split(parts, line, boost::algorithm::is_any_of(" "), boost::algorithm::token_compress_on);
                   auto parts_end = std::remove_if(parts.begin(), parts.end(),
                                                   [](const string& s) {
                                                     return s.empty();
                                                   });
                   vector<int> indices(std::distance(parts.begin(), parts_end));
                   std::transform(parts.begin(), parts_end, indices.begin(),
                                  [](const string& s) {
                                    return std::stoi(s);
                                  });
                   return indices;
                 });
  return contour_indices;
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
  google::InitGoogleLogging(argv[0]);

  if( argc < 3 ) {
    cout << "Usage: ./SingleImageReconstruction image_file pts_file" << endl;
    return -1;
  }
  string image_filename = argv[1];
  string pts_filename = argv[2];

  SingleImageReconstructor<Constraint2D> recon;
  recon.LoadModel("/home/phg/Data/Multilinear/blendshape_core.tensor");
  recon.LoadPriors("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                   "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  QImage img(image_filename.c_str());
  cout << "image size: " << img.width() << "x" << img.height() << endl;
  double image_size = img.height();
  double scale_ratio = 1.0;
  const double max_image_size = 640.0;
  scale_ratio = max_image_size / image_size;
  img = img.scaled(img.width() * scale_ratio, img.height() * scale_ratio);
  recon.SetImageSize(img.width(), img.height());
  auto landmarks = LoadIndices("/home/phg/Data/Multilinear/landmarks_73.txt");
  recon.SetIndices(landmarks);
  auto constraints = LoadConstraints(pts_filename);
  // Preprocess constraints
  for(auto& constraint : constraints) {
    constraint.data = constraint.data * scale_ratio;
    constraint.data.y = img.height() - 1 - constraint.data.y;
  }
  recon.SetConstraints(constraints);
  auto contour_indices = LoadContourIndices("/home/phg/Data/Multilinear/contourpoints.txt");
  recon.SetContourIndices(contour_indices);
  BasicMesh mesh("/home/phg/Data/Multilinear/template.obj");
  recon.SetMesh(mesh);
  recon.Reconstruct();

  auto tm = recon.GetGeometry();
  mesh.UpdateVertices(tm);
  auto R = recon.GetRotation();
  auto T = recon.GetTranslation();
  auto cam_params = recon.GetCameraParameters();

  MeshVisualizer w("reconstruction result", mesh);
  w.BindConstraints(constraints);
  w.BindImage(img);
  w.BindLandmarks(recon.GetIndices());
  w.BindUpdatedLandmarks(recon.GetUpdatedIndices());
  w.SetMeshRotationTranslation(R, T);
  w.SetCameraParameters(cam_params);
  w.resize(img.width(), img.height());
  w.show();

  return a.exec();
}
