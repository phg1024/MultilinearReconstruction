#pragma once

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

#include "ceres/ceres.h"

#include <opencv2/opencv.hpp>

#include "basicmesh.h"
#include "common.h"
#include "constraints.h"
#include "costfunctions.h"
#include "ioutilities.h"
#include "multilinearmodel.h"
#include "parameters.h"
#include "singleimagereconstructor.hpp"
#include "statsutils.h"
#include "utils.hpp"

#include "OffscreenMeshVisualizer.h"

#include "AAM/aammodel.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/program_options.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace Eigen;

po::variables_map parse_cli_args(int argc, char** argv) {
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help messages")
    ("img", po::value<string>()->required(), "Background iamge.")
    ("res", po::value<string>()->required(), "Reconstruction information.")
    ("mesh", po::value<string>()->required(), "Mesh to render.")
    ("texture", po::value<string>()->default_value(""), "Texture for the mesh.")
    ("normals", po::value<string>()->default_value(""), "Customized normals for the mesh.")
    ("settings", po::value<string>()->default_value("/home/phg/Data/Settings/mesh_vis.json"), "Rendering settings")
    ("output", po::value<string>()->required(), "Output image file.");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    return vm;
  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    exit(1);
  }
}

void VisualizeReconstructionResult(
  const string& img_filename,
  const string& res_filename,
  const string& mesh_filename,
  const string& output_image_filename,
  const map<string, string>& extra_options,
  bool scale_output=true) {

  QImage img(img_filename.c_str());
  int imgw = img.width();
  int imgh = img.height();
  if(scale_output) {
    const int target_size = 640;
    double scale = static_cast<double>(target_size) / imgw;
    imgw *= scale;
    imgh *= scale;
  }

  BasicMesh mesh(mesh_filename);
  auto recon_results = LoadReconstructionResult(res_filename);

  OffscreenMeshVisualizer visualizer(imgw, imgh);

  visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
  visualizer.SetRenderMode(OffscreenMeshVisualizer::MeshAndImage);
  visualizer.BindMesh(mesh);
  visualizer.BindImage(img);

  visualizer.SetCameraParameters(recon_results.params_cam);
  visualizer.SetMeshRotationTranslation(recon_results.params_model.R, recon_results.params_model.T);
  visualizer.SetIndexEncoded(false);
  visualizer.SetEnableLighting(true);

  if(extra_options.count("settings")) visualizer.LoadRenderingSettings(extra_options.at("settings"));
  if(extra_options.count("texture")) visualizer.BindTexture(QImage(extra_options.at("texture").c_str()));
  if(extra_options.count("normals")) {
    visualizer.SetNormals(LoadFloats(extra_options.at("normals")));
  }

  QImage output_img = visualizer.Render(true);
  output_img.save(output_image_filename.c_str());
}

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  auto vm = parse_cli_args(argc, argv);
  if(argc<5) {
    cout << "Usage: " << argv[0] << " img res mesh output" << endl;
    return 1;
  }

  VisualizeReconstructionResult(vm["img"].as<string>(),
                                vm["res"].as<string>(),
                                vm["mesh"].as<string>(),
                                vm["output"].as<string>(),
                                map<string, string>{
                                  {"normals", vm["normals"].as<string>()},
                                  {"texture", vm["texture"].as<string>()},
                                  {"settings", vm["settings"].as<string>()}
                                });
  return 0;
}
