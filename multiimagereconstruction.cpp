#include <QApplication>
#include <GL/freeglut_std.h>

#include "glog/logging.h"
#include "ioutilities.h"
#include "meshvisualizer.h"
#include "singleimagereconstructor.hpp"
#include "multiimagereconstructor.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace fs = boost::filesystem;


int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  if( argc < 2 ) {
    cout << "Usage: ./MultiImageReconstruction setting_file" << endl;
    return -1;
  }

  const string settings_filename(argv[1]);

  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");
  const string id_prior_filename("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor");
  const string exp_prior_filename("/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  const string template_mesh_filename("/home/phg/Data/Multilinear/template.obj");
  const string contour_points_filename("/home/phg/Data/Multilinear/contourpoints.txt");
  const string landmarks_filename("/home/phg/Data/Multilinear/landmarks_73.txt");


  BasicMesh mesh(template_mesh_filename);
  auto landmarks = LoadIndices(landmarks_filename);
  auto contour_indices = LoadContourIndices(contour_points_filename);


  // Create reconstructor and load the common resources
  MultiImageReconstructor<Constraint2D> recon;
  recon.LoadModel(model_filename);
  recon.LoadPriors(id_prior_filename, exp_prior_filename);
  recon.SetMesh(mesh);
  recon.SetContourIndices(contour_indices);
  recon.SetIndices(landmarks);

  // Parse the setting file and load image related resources
  fs::path settings_filepath(settings_filename);

  vector<pair<string, string>> image_points_filenames = ParseSettingsFile(settings_filename);
  for(auto& p : image_points_filenames) {
    fs::path image_filename = settings_filepath.parent_path() / fs::path(p.first);
    fs::path pts_filename = settings_filepath.parent_path() / fs::path(p.second);
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string());
    recon.AddImagePointsPair(image_points_pair);
  }

  {
    boost::timer::auto_cpu_timer t("Reconstruction finished in %w seconds.\n");
    recon.Reconstruct();
  }

  return a.exec();
}
