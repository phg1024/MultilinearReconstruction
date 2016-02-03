#include <QApplication>
#include <GL/freeglut_std.h>

#include "ioutilities.h"
#include "meshvisualizer.h"
#include "singleimagereconstructor.hpp"
#include "glog/logging.h"
#include "boost/timer/timer.hpp"

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  glutInit(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  if( argc < 3 ) {
    cout << "Usage: ./SingleImageReconstruction image_file pts_file" << endl;
    return -1;
  }

  const string image_filename(argv[1]);
  const string pts_filename(argv[2]);

  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");
  const string id_prior_filename("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor");
  const string exp_prior_filename("/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  const string template_mesh_filename("/home/phg/Data/Multilinear/template.obj");
  const string contour_points_filename("/home/phg/Data/Multilinear/contourpoints.txt");
  const string landmarks_filename("/home/phg/Data/Multilinear/landmarks_73.txt");


  BasicMesh mesh(template_mesh_filename);
  auto contour_indices = LoadContourIndices(contour_points_filename);
  auto landmarks = LoadIndices(landmarks_filename);


  // Create reconstructor and load the common resources
  SingleImageReconstructor<Constraint2D> recon;
  recon.LoadModel(model_filename);
  recon.LoadPriors(id_prior_filename, exp_prior_filename);
  recon.SetMesh(mesh);
  recon.SetContourIndices(contour_indices);
  recon.SetIndices(landmarks);

  // Load image related resources
  auto image_points_pair = LoadImageAndPoints(image_filename, pts_filename);

  QImage img = image_points_pair.first;
  auto constraints = image_points_pair.second;

  recon.SetImage(img);
  recon.SetImageSize(img.width(), img.height());
  recon.SetConstraints(constraints);

  // Do reconstruction
  {
    boost::timer::auto_cpu_timer t("Reconstruction finished in %w seconds.\n");
    recon.Reconstruct();
  }

  // Visualize reconstruction result
  auto tm = recon.GetGeometry();
  mesh.UpdateVertices(tm);
  mesh.ComputeNormals();
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

  double scale = 640.0 / img.height();
  w.resize(img.width() * scale, img.height() * scale);
  w.show();

  // Save the reconstruction results
  // w_id, w_exp, rotation, translation, camera parameters
  recon.SaveReconstructionResults(image_filename + ".res");

  return a.exec();
}
