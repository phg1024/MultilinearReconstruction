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

namespace fs = boost::filesystem;

using namespace Eigen;

void VisualizeReconstructionResult(
  const string& img_filename,
  const string& res_filename,
  const string& mesh_filename,
  const string& output_image_filename,
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

  QImage output_img = visualizer.Render(true);
  output_img.save(output_image_filename.c_str());
}

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  if(argc<5) {
    cout << "Usage: " << argv[0] << " img res mesh output" << endl;
    return 1;
  }

  VisualizeReconstructionResult(argv[1], argv[2], argv[3], argv[4]);
  return 0;
}
