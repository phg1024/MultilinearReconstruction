#include <QApplication>
#include "../OffscreenMeshVisualizer.h"
#include "../multilinearmodel.h"
#include "../basicmesh.h"
#include "../costfunctions.h"
#include "../ioutilities.h"
#include "../multilinearmodel.h"
#include "../parameters.h"
#include "../utils.hpp"

int main(int argc, char** argv) {
  QApplication a(argc, argv);

  QImage in_img(argv[1]);
  auto recon_results = LoadReconstructionResult(string(argv[1]) + ".res");

  MultilinearModel model("/home/phg/Data/Multilinear/blendshape_core.tensor");
  MultilinearModelPrior model_prior;
  model_prior.load("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor",
                   "/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");


  model.ApplyWeights(recon_results.params_model.Wid, recon_results.params_model.Wexp);
  BasicMesh mesh0("/home/phg/Data/Multilinear/template.obj");
  mesh0.UpdateVertices(model.GetTM());
  mesh0.ComputeNormals();

  float scale = 640.0 / recon_results.params_cam.image_size.y;

  OffscreenMeshVisualizer visualizer(recon_results.params_cam.image_size.x * scale,
                                     recon_results.params_cam.image_size.y * scale);

  visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
  visualizer.SetRenderMode(OffscreenMeshVisualizer::MeshAndImage);
  visualizer.BindMesh(mesh0);
  visualizer.BindImage(in_img);
  visualizer.SetCameraParameters(recon_results.params_cam);
  visualizer.SetMeshRotationTranslation(recon_results.params_model.R, recon_results.params_model.T);
  visualizer.SetIndexEncoded(false);
  visualizer.SetEnableLighting(true);

  QImage img = visualizer.Render(true);
  img.save("img.png");

  return 0;
}
