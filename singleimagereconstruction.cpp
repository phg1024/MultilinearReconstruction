#include <QApplication>
#include <GL/freeglut_std.h>

#include "ioutilities.h"
#include "meshvisualizer.h"
#include "meshvisualizer2.h"
#include "singleimagereconstructor.hpp"
#include "glog/logging.h"
#include "boost/timer/timer.hpp"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

int main(int argc, char *argv[]) {
  // program options
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help messages")
    ("img", po::value<string>()->required(), "Input image file")
    ("pts", po::value<string>()->required(), "Input points file")
    ("wid", po::value<float>(), "Initial identity weight.")
    ("dwid", po::value<float>(), "Identity weight step")
    ("wexp", po::value<float>(), "Initial expression weight")
    ("dwexp", po::value<float>(), "Expression weight step")
    ("iters", po::value<int>(), "Maximum iterations")
    ("inits", po::value<int>(), "Number of initializations")
    ("perturb_range", po::value<double>(), "Range of perturbation")
    ("error_thres", po::value<double>(), "Error threhsold")
    ("error_diff_thres", po::value<double>(), "Error difference threhsold")
    ("vis,v", "Visualize reconstruction results")
    ("no_selection", "Disable subset selection");
  po::variables_map vm;
  OptimizationParameters opt_params = OptimizationParameters::Defaults();

  string image_filename, pts_filename;
  bool visualize_results = false;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help")) {
      cout << desc << endl;
      return 1;
    }

    if(vm.count("wid")) opt_params.w_prior_id = vm["wid"].as<float>();
    if(vm.count("dwid")) opt_params.d_w_prior_id = vm["dwid"].as<float>();
    if(vm.count("wexp")) opt_params.w_prior_exp = vm["wexp"].as<float>();
    if(vm.count("dwexp")) opt_params.d_w_prior_exp = vm["dwexp"].as<float>();
    if(vm.count("iters")) opt_params.max_iters = vm["iters"].as<int>();
    if(vm.count("inits")) opt_params.num_initializations = vm["inits"].as<int>();
    if(vm.count("perturb_range")) opt_params.perturbation_range = vm["perturb_range"].as<double>();
    if(vm.count("error_thres")) opt_params.errorThreshold = vm["error_thres"].as<double>();
    if(vm.count("error_diff_thres")) opt_params.errorDiffThreshold = vm["error_diff_thres"].as<double>();
    if(vm.count("-v") || vm.count("vis")) visualize_results = true;
    image_filename = vm["img"].as<string>();
    pts_filename = vm["pts"].as<string>();

  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    return 1;
  }

  namespace fs=boost::filesystem;

  QApplication a(argc, argv);
  glutInit(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  fs::path image_path(image_filename);
  fs::path recon_path = image_path.parent_path() / "recon";

  if ( !fs::exists(image_filename) || !fs::exists(pts_filename) ){
    cout << "Either image file or points file is missing. Abort." << endl;
    return -1;
  }

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
  auto image_points_pair = LoadImageAndPoints(image_filename, pts_filename, false);

  QImage img = image_points_pair.first;
  auto constraints = image_points_pair.second;

  recon.SetImage(img);
  recon.SetImageSize(img.width(), img.height());
  recon.SetConstraints(constraints);
  recon.SetImageFilename(image_filename);

  // Do reconstruction
  {
    boost::timer::auto_cpu_timer t("Reconstruction finished in %w seconds.\n");
    recon.Reconstruct(opt_params);
  }

  // Visualize reconstruction result
  auto tm = recon.GetGeometry();
  mesh.UpdateVertices(tm);
  mesh.ComputeNormals();
  auto R = recon.GetRotation();
  auto T = recon.GetTranslation();
  auto cam_params = recon.GetCameraParameters();

  MeshVisualizer2 w("reconstruction result", mesh);
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

  {
    //QImage I(img.width(), img.height(), QImage::Format_ARGB32);
    //QPainter painter(&I);
    //w.render(&painter);
    w.paintGL();
    QImage I = w.grabFrameBuffer();
    I.save( (recon_path / image_path.filename()).string().c_str() );
  }

  if(visualize_results) {
    return a.exec();
  } else {
    return 0;
  }
}
