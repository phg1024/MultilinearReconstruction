/*
This program takes a set of blendshapes, as well as a set of initial guess of
pose and expression weights, and estimates optimal pose and expression weights.
*/
#include <QApplication>
#include <GL/freeglut_std.h>

#include "ioutilities.h"
#include "meshvisualizer.h"
#include "meshvisualizer2.h"
#include "OffscreenMeshVisualizer.h"
#include "singleimagereconstructor_exp.hpp"

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
    ("settings_file", po::value<string>()->required(), "Settings file")
    ("blendshapes_path", po::value<string>()->required(), "Input blendshapes path.")
    ("init_recon_path", po::value<string>()->required(), "Initial reconstruction parameters path.")
    ("iter", po::value<int>()->required(), "The iteration number.")
    ("wexp", po::value<float>(), "Initial expression weight")
    ("dwexp", po::value<float>(), "Expression weight step")
    ("maxiters", po::value<int>(), "Maximum iterations")
    ("inits", po::value<int>(), "Number of initializations")
    ("perturb_range", po::value<double>(), "Range of perturbation")
    ("error_thres", po::value<double>(), "Error threhsold")
    ("error_diff_thres", po::value<double>(), "Error difference threhsold")
    ("vis,v", "Visualize reconstruction results");
  po::variables_map vm;

  OptimizationParameters opt_params = OptimizationParameters::Defaults();

  string settings_filename;
  string blendshapes_path, init_recon_path;
  int iteration;
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

    settings_filename = vm["settings_file"].as<string>();
    blendshapes_path = vm["blendshapes_path"].as<string>();
    init_recon_path = vm["init_recon_path"].as<string>();
    iteration = vm["iter"].as<int>();

  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    return 1;
  }

  namespace fs=boost::filesystem;

  QApplication a(argc, argv);
  glutInit(&argc, argv);
  google::InitGoogleLogging(argv[0]);

  fs::path settings_filepath(settings_filename);

  if ( !fs::exists(settings_filename) ){
    cout << "Settings file is missing. Abort." << endl;
    return -1;
  }

  fs::path recon_path = settings_filepath.parent_path() / ("iteration_" + to_string(iteration)) / "recon";
  if(!fs::exists(recon_path)) {
    try{
      cout << "Creating blendshapes directory " << recon_path.string() << endl;
      fs::create_directory(recon_path);
    } catch(exception& e) {
      cout << e.what() << endl;
      exit(1);
    }
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
  recon.LoadBlendshapes(blendshapes_path, false);

  // Load the settings file and get all the input images and points
  vector<pair<string, string>> image_points_filenames = ParseSettingsFile(settings_filename);
  for(auto& p : image_points_filenames) {
    fs::path image_filename = settings_filepath.parent_path() / fs::path(p.first);
    fs::path pts_filename = settings_filepath.parent_path() / fs::path(p.second);
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string(), false);

    QImage img = image_points_pair.first;
    auto constraints = image_points_pair.second;

    recon.SetImage(img);
    recon.SetImageSize(img.width(), img.height());
    recon.SetConstraints(constraints);
    recon.SetImageFilename(image_filename.string());
    recon.SetOptimizationMode(
      SingleImageReconstructor<Constraint2D>::OptimizationMode(
        SingleImageReconstructor<Constraint2D>::Pose
      | SingleImageReconstructor<Constraint2D>::Expression
      | SingleImageReconstructor<Constraint2D>::FocalLength));

    // Load the initial recon results and blendshapes
    auto recon_results = LoadReconstructionResult(
      (fs::path(init_recon_path) / fs::path(p.first + ".res")).string() );

    recon.SetInitialParameters(recon_results.params_model, recon_results.params_cam);
    recon.RefereshWeights();

    // Do reconstruction
    {
      boost::timer::auto_cpu_timer t("Reconstruction finished in %w seconds.\n");
      recon.Reconstruct(opt_params);
    }

    // Visualize reconstruction result
    //auto tm = recon.GetGeometry();
    //mesh.UpdateVertices(tm);
    //mesh.ComputeNormals();
    mesh = recon.GetMesh();

    auto R = recon.GetRotation();
    auto T = recon.GetTranslation();
    auto cam_params = recon.GetCameraParameters();

    // Save the reconstruction results
    cout << "Saving results to " << (recon_path / fs::path(p.first)).string() << endl;
    // w_id, w_exp, rotation, translation, camera parameters
    recon.SaveReconstructionResults( (recon_path / fs::path(p.first)).string() + ".res" );

    {
      OffscreenMeshVisualizer visualizer(640, 640);

      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::MeshAndImage);
      visualizer.BindMesh(mesh);
      visualizer.BindImage(img);
      visualizer.SetCameraParameters(cam_params);
      visualizer.SetMeshRotationTranslation(R, T);
      visualizer.SetIndexEncoded(false);
      visualizer.SetEnableLighting(true);

      QImage I = visualizer.Render(true);
      I.save( (recon_path / p.first).string().c_str() );
    }
  }

  if(visualize_results) {
    return a.exec();
  } else {
    return 0;
  }
}
