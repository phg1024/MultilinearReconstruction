#include <QApplication>
#include <GL/freeglut_std.h>

#include "glog/logging.h"
#include "ioutilities.h"
#include "meshvisualizer.h"
#include "singleimagereconstructor.hpp"
#include "multiimagereconstructor.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/program_options.hpp"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  namespace fs = boost::filesystem;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
  ("settings_file", po::value<string>()->required(), "Input settings file")
  ("model_file", po::value<string>()->default_value("/home/phg/Data/Multilinear/blendshape_core.tensor"), "Multilinear model file")
  ("id_prior_file", po::value<string>()->default_value("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor"), "Identity prior file")
  ("exp_prior_file", po::value<string>()->default_value("/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor"), "Expression prior file")
  ("template_mesh_file", po::value<string>()->default_value("/home/phg/Data/Multilinear/template.obj"), "Template mesh file")
  ("contour_points_file", po::value<string>()->default_value("/home/phg/Data/Multilinear/contourpoints.txt"), "Contour points file")
  ("landmarks_file", po::value<string>()->default_value("/home/phg/Data/Multilinear/landmarks_73.txt"), "Landmarks file");

  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help")) {
      cout << desc << endl;
      return 1;
    }

    // nothing to do after successful parsing command line arguments

  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    return 1;
  }

  const string model_filename(vm["model_file"].as<string>());
  const string id_prior_filename(vm["id_prior_file"].as<string>());
  const string exp_prior_filename(vm["exp_prior_file"].as<string>());
  const string template_mesh_filename(vm["template_mesh_file"].as<string>());
  const string contour_points_filename(vm["contour_points_file"].as<string>());
  const string landmarks_filename(vm["landmarks_file"].as<string>());
  const string settings_filename(vm["settings_file"].as<string>());

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

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string(), false);
    recon.AddImagePointsPair(image_filename.string(), image_points_pair);
  }

  {
    boost::timer::auto_cpu_timer t("Reconstruction finished in %w seconds.\n");
    recon.Reconstruct();
  }

  //return a.exec();
  return 0;
}
