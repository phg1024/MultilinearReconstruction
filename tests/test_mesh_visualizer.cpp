#include <QApplication>
#include <QDir>
#include <GL/freeglut_std.h>
#include "../meshvisualizer.h"
#include "../multilinearmodel.h"
#include "../basicmesh.h"
#include "../costfunctions.h"
#include "../ioutilities.h"
#include "../multilinearmodel.h"
#include "../parameters.h"
#include "../utils.hpp"

int main(int argc, char** argv) {
  QApplication a(argc, argv);
  const string home_directory = QDir::homePath().toStdString();

  auto recon_results = LoadReconstructionResult(argv[1]);

  MultilinearModel model(home_directory + "/Data/Multilinear/blendshape_core.tensor");
  MultilinearModelPrior model_prior;
  model_prior.load(home_directory + "/Data/Multilinear/blendshape_u_0_aug.tensor",
                   home_directory + "/Data/Multilinear/blendshape_u_1_aug.tensor");


  model.ApplyWeights(recon_results.params_model.Wid, recon_results.params_model.Wexp);
  BasicMesh mesh0(home_directory + "/Data/Multilinear/template.obj");
  mesh0.UpdateVertices(model.GetTM());
  mesh0.ComputeNormals();

  MeshVisualizer w("template mesh", mesh0);

  w.SetMeshRotationTranslation(recon_results.params_model.R, recon_results.params_model.T);
  w.SetCameraParameters(recon_results.params_cam);

  float scale = 640.0 / recon_results.params_cam.image_size.y;
  w.resize(recon_results.params_cam.image_size.x * scale, recon_results.params_cam.image_size.y * scale);

  w.show();
  return a.exec();
}
