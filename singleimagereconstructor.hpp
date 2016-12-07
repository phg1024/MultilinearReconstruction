#ifndef MULTILINEARRECONSTRUCTOR_HPP
#define MULTILINEARRECONSTRUCTOR_HPP

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

using namespace Eigen;

#include "ceres/ceres.h"

#include "basicmesh.h"
#include "common.h"
#include "constraints.h"
#include "costfunctions.h"
#include "multilinearmodel.h"
#include "parameters.h"
#include "statsutils.h"
#include "utils.hpp"

#include "boost/timer/timer.hpp"

#include <opencv2/opencv.hpp>

#include "glm/ext.hpp"
#include "glm/gtx/norm.hpp"

#include <QTest>

#define USE_ANALYTIC_COST_FUNCTIONS 1

static double REFERENCE_SCALE = 1.0;

template<typename Constraint>
class SingleImageReconstructor {
public:
  enum OptimizationMode {
    Pose = 0x1,
    Identity = 0x2,
    Expression = 0x4,
    FocalLength = 0x8,
    All = 0xf
  };

  SingleImageReconstructor()
    : opt_mode(All), need_precise_result(false), is_parameters_initialized(false), display_step_result(false) {}

  void LoadModel(const string &filename) { model = MultilinearModel(filename); }

  void LoadPriors(const string &filename_id, const string &filename_exp) {
    prior.load(filename_id, filename_exp);
  }

  void SetContourIndices(
    const vector<vector<int>> &contour_points) { contour_indices = contour_points; }

  void SetConstraints(
    const vector<Constraint> &cons) { params_recon.cons = cons; }

  void SetImage(const QImage& img_in) {
    img = img_in;
  }

  void SetImageSize(int w, int h) {
    params_recon.imageWidth = w;
    params_recon.imageHeight = h;
  }

  void SetMesh(const BasicMesh &mesh_in) {
    mesh = mesh_in;
  }

  const BasicMesh& GetMesh() const {
    return mesh;
  }

  void SetOptimizationParameters(const OptimizationParameters &params) {
    params_opt = params;
  }

  void SetInitialParameters(const ModelParameters& model_params,
                            const CameraParameters& camera_params);

  void SetOptimizationMode(OptimizationMode mode) {
    opt_mode = mode;
  }

  bool Reconstruct(OptimizationParameters params = OptimizationParameters::Defaults());

  const ModelParameters &GetModelParameters() const { return params_model; }

  void SetModelParameters(const ModelParameters& params) { params_model = params; }

  void SetIdentityPrior(const VectorXd& mu_id) {
    prior.Wid0 = mu_id;
  }

  const Vector3d &GetRotation() const { return params_model.R; }

  const Vector3d &GetTranslation() const { return params_model.T; }

  const VectorXd &GetIdentityWeights() const { return params_model.Wid; }

  const VectorXd &GetExpressionWeights() const { return params_model.Wexp_FACS; }

  const Tensor1 &GetGeometry() const { return model.GetTM(); }

  const CameraParameters &GetCameraParameters() const { return params_cam; }

  void SetCameraParameters(const CameraParameters& params) { params_cam = params; }

  const vector<int> GetIndices() const { return indices; }

  void SetIndices(const vector<int> &indices_vec) { indices = indices_vec; }

  void SetImageFilename(const string& image_filename_in) {
    image_filename = image_filename_in;
  }

  vector<int> GetUpdatedIndices() const {
    vector<int> idxs;
    for (size_t i = 0; i < params_recon.cons.size(); ++i) {
      idxs.push_back(params_recon.cons[i].vidx);
    }
    return idxs;
  }

  const ReconstructionStats GetStats() const {
    return recon_stats;
  }

  void SaveReconstructionResults(const string& filename) const {
    ofstream fout(filename);
    fout << params_cam << "\n";
    fout << params_model << "\n";
    fout << recon_stats << endl;
    fout.close();
  }

  void ToggleDisplayStepResult() {
    display_step_result = !display_step_result;
  }

protected:
  void InitializeParameters(bool with_perturbation=false, double perturb_range=0.0);

  void UpdateModels();

  void ProcrustesAnalysis();

  void OptimizeForPosition();

  void OptimizeForPose(int iteration);

  void OptimizeForPose_opencv(int iteration);

  void OptimizeForFocalLength();

  void OptimizeForExpression(int iteration);

  void OptimizeForExpression_FACS(int iteration);

  void OptimizeForIdentity(int iteration);

  void UpdateContourIndices(int iteration);

  double ComputeError();

private:
  MultilinearModel model;
  vector<MultilinearModel> model_projected;
  MultilinearModelPrior prior;
  vector<vector<int>> contour_indices;

  vector<int> indices;
  BasicMesh mesh;

  QImage img;
  string image_filename;

  CameraParameters params_cam;
  ModelParameters params_model;
  ReconstructionParameters<Constraint> params_recon;
  OptimizationParameters params_opt;
  ReconstructionStats recon_stats;

  OptimizationMode opt_mode;

  bool need_precise_result;
  bool is_parameters_initialized;
  bool display_step_result;
};

template <typename Constraint>
void SingleImageReconstructor<Constraint>::SetInitialParameters(
  const ModelParameters& model_params, const CameraParameters& camera_params) {
  SetModelParameters(model_params);
  SetCameraParameters(camera_params);
  UpdateModels();
  is_parameters_initialized = true;
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::InitializeParameters(bool with_perturbation, double perturb_range) {
  boost::timer::auto_cpu_timer timer(
    "Parameters initialization time = %w seconds.\n");

  const int num_contour_points = 15;

  // Initialization camera parameters, model parameters and projected models

  // Camera parameters
  // Typical camera fov for 50mm cameras
  CameraParameters camera_params = CameraParameters::DefaultParameters(
    params_recon.imageWidth, params_recon.imageHeight);

  // Model parameters
  ModelParameters model_params = ModelParameters::DefaultParameters(prior.Uid,
                                                                    prior.Uexp);

  if(with_perturbation) {
    // change the identity weights and the experssion weights a little
    const double range = 0.05;
    model_params.Wid = StatsUtils::perturb(model_params.Wid, perturb_range, prior.sigma_Wid);
    model_params.Wexp_FACS = StatsUtils::perturb(model_params.Wexp_FACS, perturb_range);
    model_params.Wexp_FACS(1) = 1.0;
    model_params.Wexp = model_params.Wexp_FACS.transpose() * prior.Uexp;
  }

  // No rotation and translation
  model_params.R = Vector3d(1e-3, 1e-3, 1e-3);
  model_params.T = Vector3d(0, 0, -1.0);

  SetInitialParameters(model_params, camera_params);
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::UpdateModels() {
  model.ApplyWeights(params_model.Wid, params_model.Wexp);

  for (size_t i = 0; i < indices.size(); ++i) {
    params_recon.cons[i].vidx = indices[i];
    params_recon.cons[i].weight = 1.0;

    params_model.vindices(i) = indices[i];
  }

  // Create initial projected models
  model_projected.resize(params_recon.cons.size());
  for (size_t i = 0; i < params_recon.cons.size(); ++i) {
    model_projected[i] = model.project(vector<int>(1, indices[i]));
    model_projected[i].ApplyWeights(params_model.Wid, params_model.Wexp);
  }
}

template<typename Constraint>
bool SingleImageReconstructor<Constraint>::Reconstruct(OptimizationParameters opt_params) {
  // Initialize parameters
  cout << "Reconstruction begins." << endl;

  bool iterative_recon_converged = false;
  int iterative_recon_run_i = 0;

  MatrixXd wid_history(opt_params.num_initializations, 50);

  while(true) {

    // Initialize model parameters
    if(iterative_recon_run_i == 0) {
      if(!is_parameters_initialized) {
        InitializeParameters(false);
      }
    } else {
      VectorXd wid_init = params_model.Wid;

      // use the mean from the previous round as initial parameters
      params_model.Wid = StatsUtils::mean(wid_history);

      double wid_diff = (params_model.Wid - wid_init).norm();
      ColorStream(ColorOutput::Red) << "wid_diff = " << wid_diff;
      if(wid_diff < opt_params.errorThreshold) break;
    }

    VectorXd wid0 = params_model.Wid;
    for(int run_i = 0; run_i < opt_params.num_initializations; ++run_i) {
      const int num_contour_points = 15;

      // Reconstruction begins
      params_model.Wid = StatsUtils::perturb(wid0, opt_params.perturbation_range, prior.sigma_Wid);

      ColorStream(ColorOutput::Red) << "Initial Error = " << ComputeError();

      // Optimization parameters
      const int kMaxIterations = opt_params.max_iters;
      const double init_weights = 1.0;
      prior.weight_Wid = opt_params.w_prior_id;
      const double d_wid = opt_params.d_w_prior_id;
      prior.weight_Wexp = opt_params.w_prior_exp;
      const double d_wexp = opt_params.d_w_prior_exp;
      int iters = 0;

      // Before entering the main loop, estimate the translation and roataion around z-axis first
      ProcrustesAnalysis();

      for (int i = 0; i < num_contour_points; ++i) {
        params_recon.cons[i].weight = 0.5;
      }
      OptimizeForPosition();
      for (int i = 0; i < num_contour_points; ++i) {
        params_recon.cons[i].weight = 1.0;
      }

      while (iters++ < kMaxIterations) {
        ColorStream(ColorOutput::Green) << "Iteration " << iters << " begins.";
        {
          boost::timer::auto_cpu_timer timer_loop(
            "[Main loop] Iteration time = %w seconds.\n");

          if((opt_mode & (Identity | Expression))){
            boost::timer::auto_cpu_timer timer(
              "[Main loop] Multilinear model weights update time = %w seconds.\n");
            //model.ApplyWeights(params_model.Wid, params_model.Wexp);
            model.UpdateTM0(params_model.Wid);
            model.UpdateTMWithTM1(params_model.Wid);
          }
          mesh.UpdateVertices(model.GetTM());
          mesh.ComputeNormals();

          if(opt_mode & Pose) {
            for (int pose_opt_iter = 0; pose_opt_iter < 1; ++pose_opt_iter) {
              OptimizeForPose(iters);
              UpdateContourIndices(iters);
            }
          }

          if(opt_mode & Expression) {
            //OptimizeForExpression(iters*100);
            OptimizeForExpression_FACS(iters*10);
          }

          if(opt_mode & FocalLength) {
            OptimizeForFocalLength();
          }

          if(opt_mode & Expression){
            boost::timer::auto_cpu_timer timer(
              "[Main loop] Multilinear model weights update time = %w seconds.\n");
            //model.ApplyWeights(params_model.Wid, params_model.Wexp);
            model.UpdateTM1(params_model.Wexp);
            model.UpdateTMWithTM0(params_model.Wexp);
          }
          mesh.UpdateVertices(model.GetTM());
          mesh.ComputeNormals();

          if(opt_mode & Pose) {
            for (int pose_opt_iter = 0; pose_opt_iter < 1; ++pose_opt_iter) {
              OptimizeForPose(iters);
              UpdateContourIndices(iters);
            }
          }

          if(opt_mode & Identity) {
            OptimizeForIdentity(iters*10);
          }

          if(opt_mode & FocalLength) {
            OptimizeForFocalLength();
          }

          double E = ComputeError();

          ColorStream(ColorOutput::Red) << "Iteration " << iters << " Error = " <<
          E;

          // Adjust weights
          prior.weight_Wid /= d_wid; prior.weight_Wid = max(prior.weight_Wid, 1.0);
          prior.weight_Wexp /= d_wexp; prior.weight_Wexp = max(prior.weight_Wexp, 1.0);
          for (int i = 0; i < num_contour_points; ++i) {
            params_recon.cons[i].weight = sqrt(params_recon.cons[i].weight);
          }
        }
        ColorStream(ColorOutput::Green) << "Iteration " << iters << " finished.";

        // Visualize reconstruction result
        if(display_step_result) {
          auto tm = GetGeometry();
          mesh.UpdateVertices(tm);
          auto R = GetRotation();
          auto T = GetTranslation();
          auto cam_params = GetCameraParameters();

          MeshVisualizer *w = new MeshVisualizer("reconstruction result " + std::to_string(iters), mesh);
          w->BindConstraints(params_recon.cons);
          w->BindImage(img);
          w->BindLandmarks(GetIndices());
          w->BindUpdatedLandmarks(GetUpdatedIndices());
          w->SetMeshRotationTranslation(R, T);
          w->SetCameraParameters(cam_params);

          double scale = 640.0 / params_cam.image_size.y;
          w->resize(params_cam.image_size.x * scale, params_cam.image_size.y * scale);
          w->show();
        }
      }

      cout << "Reconstruction done." << endl;
      model.ApplyWeights(params_model.Wid, params_model.Wexp);

      //SaveReconstructionResults(image_filename + "_run_"+ to_string(run_i) + ".res");

      wid_history.row(run_i) = params_model.Wid;
    }

    ++iterative_recon_run_i;
  }

  return true;
}

template<typename Constraint>
double SingleImageReconstructor<Constraint>::ComputeError() {
  boost::timer::auto_cpu_timer timer_all(
    "[Error computation] Error computation time = %w seconds.\n");

  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1],
                                 params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0],
                                              params_model.T[1],
                                              params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  const double puple_distance = glm::distance(
    0.5 * (params_recon.cons[28].data + params_recon.cons[30].data),
    0.5 * (params_recon.cons[32].data + params_recon.cons[34].data));

  double E = 0;
  double max_error = 0, min_error = 1e9;
  for (size_t i = 0; i < indices.size(); ++i) {
    auto &model_i = model_projected[i];
    //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
    auto tm = model_i.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    auto q = ProjectPoint(p, Mview, params_cam);
    double dx = q.x - params_recon.cons[i].data.x;
    double dy = q.y - params_recon.cons[i].data.y;
    double error_i = sqrt(dx * dx + dy * dy) / puple_distance;
    max_error = max(max_error, error_i);
    min_error = min(min_error, error_i);
    E += error_i;
  }

  recon_stats.max_error = max_error;
  recon_stats.min_error = min_error;
  recon_stats.avg_error = E / indices.size();

  return E / indices.size();
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::ProcrustesAnalysis() {
  boost::timer::auto_cpu_timer timer_all(
    "[Position optimization] Total time = %w seconds.\n");
  const int N = indices.size();

  // normalize the constraints
  glm::dvec2 mean_q(0, 0);
  for (int i = 0; i < N; ++i) {
    mean_q += params_recon.cons[i].data;
  }
  mean_q /= N;

  vector<glm::dvec2> qi2d(N);
  double scale_q = 0.0;
  for(int i=0;i<N;++i) {
    qi2d[i] = params_recon.cons[i].data - mean_q;
    scale_q += glm::dot(qi2d[i], qi2d[i]);
  }
  scale_q /= N;
  for(auto& q : qi2d) q /= scale_q;

  // normalize the points on the mesh
  vector<glm::dvec2> v(N);
  glm::dvec2 mean_v(0, 0);
  for(int i=0;i<N;++i) {
    auto &model_i = model_projected[i];
    auto tm = model_i.GetTM();
    v[i] = glm::dvec2(tm[0], tm[1]);
    mean_v += v[i];
  }
  mean_v /= N;

  vector<glm::dvec2> vi2d(N);
  double scale_v = 0.0;
  for(int i=0;i<N;++i) {
    vi2d[i] = v[i] - mean_v;
    scale_v += glm::dot(vi2d[i], vi2d[i]);
  }
  scale_v /= N;
  for(auto& vi : vi2d) vi /= scale_v;

  double denom = 0.0, numer = 0.0;
  for(int i=0;i<N;++i) {
    denom += glm::dot(vi2d[i], qi2d[i]);
    numer += glm::dot(vi2d[i], glm::dvec2(qi2d[i].y, -qi2d[i].x));
  }
  double theta2d = atan2(numer, denom);

  params_model.R[2] = theta2d;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForPosition() {
  boost::timer::auto_cpu_timer timer_all(
    "[Position optimization] Total time = %w seconds.\n");

  ceres::Problem problem;
  vector<double> params{params_model.T[0], params_model.T[1],
                        params_model.T[2]};

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Position optimization] Problem construction time = %w seconds.\n");

    for (size_t i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
#if USE_ANALYTIC_COST_FUNCTIONS
      ceres::CostFunction *cost_function = new PositionCostFunction_analytic(
        model_i,
        params_recon.cons[i],
        params_cam,
        params_model.R[2]);
#else
      ceres::CostFunction *cost_function =
        new ceres::NumericDiffCostFunction<PositionCostFunction, ceres::CENTRAL, 1, 3>(
          new PositionCostFunction(model_i,
                                   params_recon.cons[i],
                                   params_cam,
                                   params_model.R[2]));
#endif
      problem.AddResidualBlock(cost_function, NULL, params.data());
    }
  }

  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Position optimization] Problem solve time = %w seconds.\n");

    const int max_tries = 5;
    for(int i=0;i<max_tries;++i) {
      ceres::Solver::Options options;
      options.max_num_iterations = 100;
      DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      DEBUG_OUTPUT(summary.BriefReport());
      //cout << params[0] << ' ' << params[1] << ' ' << params[2] << endl;
      if(i == max_tries - 1) break;
      params[0] += (rand() % 128) / 128.0;
      params[1] += (rand() % 128) / 128.0;
      params[2] += (rand() % 128) / 128.0;
    }
  }

  Vector3d newT(params[0], params[1], params[2]);

  DEBUG_OUTPUT(
    "T: " << params_model.T.transpose() << " -> " << newT.transpose());

  params_model.T = newT;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForPose_opencv(int iteration) {
  boost::timer::auto_cpu_timer timer_all(
    "[Pose optimization] Total time = %w seconds.\n");

  glm::dmat4 projection_matrix;
  glm::dmat4 rotation_matrix;
  glm::dvec3 translation_vector;

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Pose optimization] Problem construction time = %w seconds.\n");

    vector<cv::Point3f> mesh_points;
    vector<cv::Point2f> image_points;
    for (int i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      auto tm = model_i.GetTM();
      mesh_points.push_back(cv::Point3f(tm[0], tm[1], tm[2]));
      image_points.push_back(cv::Point2f(params_recon.cons[i].data.x,
                                         params_recon.cons[i].data.y));
    }

    cv::Mat mp(mesh_points);
    cv::Mat ip(image_points);

    params_cam.focal_length = 1000.0;
    double _cm[9] = {params_cam.focal_length, 0, params_cam.image_size.x*0.5,
                     0, params_cam.focal_length, params_cam.image_size.y*0.5,
                     0,  0,  1};
    cv::Mat camMatrix = cv::Mat(3, 3, CV_64FC1, _cm);
    double _dc[] = {0, 0, 0, 0};

    const double far = 1000.0, near = 0.01;
#if 0
    projection_matrix = glm::dmat4(params_cam.focal_length, 0, 0, 0,
                                   0, params_cam.focal_length, 0, 0,
                                   params_cam.image_size.x*0.5, params_cam.image_size.y*0.5, -(far+near)/(far-near), -1,
                                   0,  0, -2.0*far*near/(far-near), 0);
#else
    projection_matrix = glm::dmat4(-params_cam.focal_length / (0.5 * params_cam.image_size.x), 0, 0, 0,
                                   0, -params_cam.focal_length / (0.5 * params_cam.image_size.y), 0, 0,
                                   0, 0, -(far+near)/(far-near), -1,
                                   0, 0, -2.0*far*near/(far-near), 0);
#endif
    cout << glm::to_string(projection_matrix) << endl;

    vector<double> rv(3), tv(3);
    cv::Mat rvec = cv::Mat(rv);

    auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1], params_model.R[2]);
    double _d[9] = {Rmat[0][0], Rmat[1][0], Rmat[2][0],
                    Rmat[0][1], Rmat[1][1], Rmat[2][1],
                    Rmat[0][2], Rmat[1][2], Rmat[2][2]}; //rotation: looking at -z axis
    cv::Mat Rmat0 = cv::Mat(3, 3, CV_64FC1, _d);
    cout << Rmat0 << endl;
    cv::Rodrigues(Rmat0, rvec);

    tv[0] = params_model.T[0]; tv[1] = params_model.T[1]; tv[2] = params_model.T[2];
    cv::Mat tvec = cv::Mat(tv);

    cv::solvePnP(mp, ip, camMatrix, cv::Mat(1, 4, CV_64FC1, _dc), rvec, tvec, false);

    cv::Mat rmat(3, 3, CV_64FC1);
    cv::Rodrigues(rvec, rmat);

    double* _r = rmat.ptr<double>();
  	printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
  		_r[0],_r[1],_r[2],_r[3],_r[4],_r[5],_r[6],_r[7],_r[8]);
    double* _t = tvec.ptr<double>();
  	printf("trans vec: \n %.3f %.3f %.3f\n", _t[0], _t[1], _t[2]);

    // rotation and translation
    double _pm[12] = {_r[0],_r[1],_r[2], _t[0],
  					          _r[3],_r[4],_r[5], _t[1],
  					          _r[6],_r[7],_r[8], _t[2]};

  	cv::Matx34d P(_pm);
  	cv::Mat KP = camMatrix * cv::Mat(P);

    translation_vector = glm::dvec3(_t[0], _t[1], _t[2]);
    rotation_matrix = glm::dmat4(_r[0], _r[3], _r[6], 0,
                                 _r[1], _r[4], _r[7], 0,
                                 _r[2], _r[5], _r[8], 0,
                                 0,     0,     0, 1);

    glm::dmat4 view_matrix = glm::lookAt(glm::dvec3(0, 0, 0),
                                         glm::dvec3(0, 0, -1),
                                         glm::dvec3(0, 1, 0));

    view_matrix = glm::dmat4(1.0);

  	//reproject object points - check validity of found projection matrix
  	for (int i=0; i<mp.rows; i++) {
  		cv::Mat_<double> X = (cv::Mat_<double>(4,1) << mp.at<float>(i,0),mp.at<float>(i,1),mp.at<float>(i,2),1.0);
  		cout << "point #" << i << ": " << mesh_points[i] << " -> ";
  		cv::Mat_<double> opt_p = KP * X;
  		cv::Point2f opt_p_img(opt_p(0)/opt_p(2),opt_p(1)/opt_p(2));
  		cout << opt_p_img << " vs " << image_points[i] << endl;
      glm::dvec4 p_opengl = projection_matrix * view_matrix * glm::translate(glm::dmat4(1.0), translation_vector) * rotation_matrix
                          * glm::dvec4(mp.at<float>(i,0), mp.at<float>(i,1), mp.at<float>(i,2), 1.0);
      p_opengl = p_opengl / p_opengl.w;
      p_opengl.x = (p_opengl.x + 1.0) * params_cam.image_size.x * 0.5;
      p_opengl.y = (p_opengl.y + 1.0) * params_cam.image_size.y * 0.5;
      p_opengl.z = (p_opengl.z + 1.0) * 0.5;
      cout << p_opengl.x << ' ' << p_opengl.y << ' ' << p_opengl.z << ' ' << p_opengl.w << endl;
      glm::dvec3 q_opengl = glm::project(glm::dvec3(mp.at<float>(i,0), mp.at<float>(i,1), mp.at<float>(i,2)),
                                         view_matrix * glm::translate(glm::dmat4(1.0), translation_vector) * rotation_matrix,
                                         projection_matrix, glm::ivec4(0, 0, params_cam.image_size.x, params_cam.image_size.y));
      cout << q_opengl.x << ' ' << q_opengl.y << ' ' << q_opengl.z << endl;
  	}
    cout << params_cam.image_size.x << ' ' << params_cam.image_size.y << endl;

    cv::Mat mtxR, mtxQ, Qx, Qy, Qz;
    cv::RQDecomp3x3(rmat, mtxR, mtxQ, Qx, Qy, Qz);

    cout << Qx << endl;
    cout << Qy << endl;
    cout << Qz << endl;

    params_model.T[0] = _t[0];
    params_model.T[1] = _t[1];
    params_model.T[2] = _t[2];

    params_model.R[0] = acos(Qy.ptr<double>()[0]);
    params_model.R[1] = acos(Qx.ptr<double>()[4]);
    params_model.R[2] = acos(Qz.ptr<double>()[0]);
  }

  // Now visualize the result
  if(1) {
    auto tm = GetGeometry();
    mesh.UpdateVertices(tm);
    auto R = GetRotation();
    auto T = GetTranslation();
    auto cam_params = GetCameraParameters();

    MeshVisualizer *w = new MeshVisualizer("pose estimation", mesh);
    w->BindConstraints(params_recon.cons);
    w->BindImage(img);
    w->BindLandmarks(GetIndices());
    w->BindUpdatedLandmarks(GetUpdatedIndices());

    w->SetMeshRotationTranslation(R, T);
    w->SetCameraParameters(cam_params);

    w->SetRotationMatrixTranslationVector(rotation_matrix, translation_vector);

    w->resize(params_cam.image_size.x, params_cam.image_size.y);
    w->show();

    QTest::qWait(100000);
  }
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForPose(int iteration) {
  boost::timer::auto_cpu_timer timer_all(
    "[Pose optimization] Total time = %w seconds.\n");

  ceres::Problem problem;
  vector<double> params{params_model.R[0], params_model.R[1], params_model.R[2], params_model.T[0], params_model.T[1], params_model.T[2]};

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Pose optimization] Problem construction time = %w seconds.\n");

    for (size_t i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
      Constraint2D cons_i = params_recon.cons[i];
      if(i<15) cons_i.weight = 0.3 * iteration;
      else if(i>45 && i<64) cons_i.weight = 0.3 * iteration;
      else cons_i.weight = 1.0;

#if USE_ANALYTIC_COST_FUNCTIONS
      ceres::CostFunction *cost_function =
        new PoseCostFunction_analytic(model_i, cons_i,
                                      params_cam);
      problem.AddResidualBlock(cost_function, NULL, params.data(),
                               params.data() + 3);
#else
      ceres::CostFunction *cost_function =
        new ceres::NumericDiffCostFunction<PoseCostFunction, ceres::CENTRAL, 1, 6>(
          new PoseCostFunction(model_i,
                               cons_i,
                               params_cam));
      problem.AddResidualBlock(cost_function, NULL, params.data());
#endif
    }

#if 1
    // Add a regularization term
    ceres::CostFunction *reg_cost_function =
      new ceres::NumericDiffCostFunction<PoseRegularizationTerm, ceres::CENTRAL, 1, 3>(
        new PoseRegularizationTerm(1.0)
      );
    problem.AddResidualBlock(reg_cost_function, NULL, params.data());
#endif
  }

  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Pose optimization] Problem solve time = %w seconds.\n");

    ceres::Solver::Options options;
    options.max_num_iterations = 15;

    options.num_threads = 8;
    options.num_linear_solver_threads = 8;

    //options.minimizer_type = ceres::LINE_SEARCH;
    //options.line_search_direction_type = ceres::LBFGS;

    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport())
  }

  Vector3d newR(params[0], params[1], params[2]);
  Vector3d newT(params[3], params[4], params[5]);
  DEBUG_OUTPUT(
    "R: " << params_model.R.transpose() << " -> " << newR.transpose())
  DEBUG_OUTPUT(
    "T: " << params_model.T.transpose() << " -> " << newT.transpose())
  params_model.R = newR;
  params_model.T = newT;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForFocalLength() {
  boost::timer::auto_cpu_timer timer_all(
    "[Focal length optimization] Total time = %w seconds.\n");

  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1],
                                 params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0],
                                              params_model.T[1],
                                              params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  // Create projection matrix
  const double aspect_ratio =
    params_cam.image_size.x / params_cam.image_size.y;

  const double far = params_cam.far;
  // near is the focal length
  const double near = params_cam.focal_length;
  const double top = near * tan(params_cam.fovy * 0.5);
  const double right = top * aspect_ratio;

  double numer = 0.0, denom = 0.0;
  const double sx = params_cam.image_size.x, sy = params_cam.image_size.y;
  for (size_t i = 0; i < indices.size(); ++i) {
    auto &model_i = model_projected[i];
    // Must apply weights here because the weights are just updated
    model_i.ApplyWeights(params_model.Wid, params_model.Wexp);

    auto tm = model_i.GetTM();
    glm::dvec4 p(tm[0], tm[1], tm[2], 1.0);
    auto P = Mview * p;

    double x_z = P.x / P.z;
    double y_z = P.y / P.z;

    double xi = params_recon.cons[i].data.x;
    double yi = params_recon.cons[i].data.y;

    numer += (sx - 2 * xi) * x_z + (sy - 2 * yi) * y_z;
    denom += sy * (x_z * x_z + y_z * y_z);
  }
  double new_f = numer / denom;
  DEBUG_OUTPUT("focal length: " << params_cam.focal_length << " -> " << new_f)
  params_cam.focal_length = new_f;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForExpression(
  int iteration) {
  boost::timer::auto_cpu_timer timer_all(
    "[Expression optimization] Total time = %w seconds.\n");

  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1],
                                 params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0],
                                              params_model.T[1],
                                              params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  double puple_distance = glm::distance(
    0.5 * (params_recon.cons[28].data + params_recon.cons[30].data),
    0.5 * (params_recon.cons[32].data + params_recon.cons[34].data));
  double prior_scale = REFERENCE_SCALE / puple_distance;

  // Define the optimization problem
  ceres::Problem problem;
  VectorXd params = params_model.Wexp;

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Expression optimization] Problem construction time = %w seconds.\n");
    for (int i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
      ceres::DynamicNumericDiffCostFunction<ExpressionCostFunction> *cost_function =
        new ceres::DynamicNumericDiffCostFunction<ExpressionCostFunction>(
          new ExpressionCostFunction(model_i,
                                     params_recon.cons[i],
                                     params.size(),
                                     Mview,
                                     params_cam));
      cost_function->AddParameterBlock(params.size());
      cost_function->SetNumResiduals(1);
      problem.AddResidualBlock(cost_function, NULL, params.data());
    }

    ceres::DynamicNumericDiffCostFunction<PriorCostFunction> *prior_cost_function =
      new ceres::DynamicNumericDiffCostFunction<PriorCostFunction>(
        new PriorCostFunction(prior.Wexp_avg, prior.inv_sigma_Wexp,
                              prior.weight_Wexp * prior_scale));
    prior_cost_function->AddParameterBlock(params.size());
    prior_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(prior_cost_function, NULL, params.data());

    for(int i=0;i<params.size();++i) {
      problem.SetParameterLowerBound(params.data(), i, 0.0);
      problem.SetParameterUpperBound(params.data(), i, 1.0);
    }
  }

  // Solve it
  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Expression optimization] Problem solve time = %w seconds.\n");
    ceres::Solver::Options options;
    options.max_num_iterations = iteration * 3;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::STEEPEST_DESCENT;
    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport())

    options.max_num_iterations = iteration * 5;
    options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    ceres::Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport())
  }

  // Update the model parameters
  DEBUG_OUTPUT(params_model.Wexp.transpose() << endl
               << " -> " << endl
               << params.transpose())
  params_model.Wexp = params;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForExpression_FACS(
  int iteration) {
  boost::timer::auto_cpu_timer timer_all(
    "[Expression optimization] Total time = %w seconds.\n");
  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1],
                                 params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0],
                                              params_model.T[1],
                                              params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  double puple_distance = glm::distance(
    0.5 * (params_recon.cons[28].data + params_recon.cons[30].data),
    0.5 * (params_recon.cons[32].data + params_recon.cons[34].data));
  double prior_scale = REFERENCE_SCALE / puple_distance;

  // Define the optimization problem
  ceres::Problem problem;

  VectorXd params = params_model.Wexp_FACS;

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Expression optimization] Problem construction time = %w seconds.\n");
    for (size_t i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
#if USE_ANALYTIC_COST_FUNCTIONS
      ceres::CostFunction *cost_function = new ExpressionCostFunction_FACS_analytic(
        model_i, params_recon.cons[i], params.size(), Mview, Rmat, prior.Uexp,
        params_cam);
#else
      ceres::DynamicNumericDiffCostFunction<ExpressionCostFunction_FACS> *cost_function =
        new ceres::DynamicNumericDiffCostFunction<ExpressionCostFunction_FACS>(
          new ExpressionCostFunction_FACS(model_i,
                                          params_recon.cons[i],
                                          params.size(),
                                          Mview,
                                          prior.Uexp,
                                          params_cam));
      // Optimize the last 46 weights only
      cost_function->AddParameterBlock(params.size() - 1);
      cost_function->SetNumResiduals(1);
#endif
      // Optimize the last 46 weights only
      problem.AddResidualBlock(cost_function, NULL, params.data() + 1);
    }

    // Expression prior term
    ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationCostFunction> *prior_cost_function =
      new ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationCostFunction>(
        new ExpressionRegularizationCostFunction(prior.Wexp_avg,
                                                 prior.inv_sigma_Wexp,
                                                 prior.Uexp, prior.weight_Wexp *
                                                             prior_scale));
    prior_cost_function->AddParameterBlock(params.size()-1);
    prior_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(prior_cost_function, NULL, params.data()+1);

    // Expression regularization term, minimize the norm of the expression vector
    const double reg_factor = 100.0 / puple_distance;
    ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationTerm> *reg_cost_function =
      new ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationTerm>(
        new ExpressionRegularizationTerm(10.0 / reg_factor * exp(-(iteration / 10 - 1) * 0.25))
      );
    reg_cost_function->AddParameterBlock(params.size()-1);
    reg_cost_function->SetNumResiduals(params.size()-1);
    problem.AddResidualBlock(reg_cost_function, NULL, params.data()+1);

    for(int i=0;i<params.size()-1;++i) {
      problem.SetParameterLowerBound(params.data()+1, i, 0.0);
      problem.SetParameterUpperBound(params.data()+1, i, 1.0);
    }
  }

  // Solve it
  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Expression optimization] Problem solve time = %w seconds.\n");
    ceres::Solver::Options options;
    options.max_num_iterations = iteration;

    options.num_threads = 8;
    options.num_linear_solver_threads = 8;

#if 1
    options.initial_trust_region_radius = 1.0;
    options.min_trust_region_radius = 0.5;
    options.max_trust_region_radius = 2.0;
    options.min_lm_diagonal = 1.0;
    options.max_lm_diagonal = 1.0;
#else
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::LBFGS;
#endif

    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport())

    //if (need_precise_result)
    {
      options.max_num_iterations = 2;
      options.minimizer_type = ceres::LINE_SEARCH;
      options.line_search_direction_type = ceres::LBFGS;
      ceres::Solve(options, &problem, &summary);
      DEBUG_OUTPUT(summary.BriefReport())
    }
  }

  // Update the model parameters
  DEBUG_OUTPUT(params_model.Wexp_FACS.transpose() << endl
               << " -> " << endl
               << params.transpose())
  params_model.Wexp_FACS = params;

  // Post process the FACS weights
  VectorXd weights_exp = params;
  weights_exp[0] = 1.0 - params.bottomRows(46).sum();
  params_model.Wexp = weights_exp.transpose() * prior.Uexp;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForIdentity(int iteration) {
  boost::timer::auto_cpu_timer timer_all(
    "[Identity optimization] Total time = %w seconds.\n");

  // Create view matrix
  glm::dmat4 Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1],
                                       params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0],
                                              params_model.T[1],
                                              params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  double puple_distance = glm::distance(
    0.5 * (params_recon.cons[28].data + params_recon.cons[30].data),
    0.5 * (params_recon.cons[32].data + params_recon.cons[34].data));
  double prior_scale = REFERENCE_SCALE / puple_distance;

  // Define the optimization problem
  ceres::Problem problem;
  VectorXd params = params_model.Wid;

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Identity optimization] Problem construction time = %w seconds.\n");
    for (size_t i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);

#if USE_ANALYTIC_COST_FUNCTIONS
      ceres::CostFunction *cost_function = new IdentityCostFunction_analytic(
        model_i, params_recon.cons[i], params.size(), Mview, Rmat, params_cam);
#else
      ceres::DynamicNumericDiffCostFunction<IdentityCostFunction> *cost_function =
        new ceres::DynamicNumericDiffCostFunction<IdentityCostFunction>(
          new IdentityCostFunction(model_i, params_recon.cons[i], params.size(),
                                   Mview, params_cam));

      cost_function->AddParameterBlock(params.size());
      cost_function->SetNumResiduals(1);
#endif
      problem.AddResidualBlock(cost_function, NULL, params.data());
    }

    // Prior term
    #if 0
    ceres::DynamicNumericDiffCostFunction<PriorCostFunction> *prior_cost_function =
      new ceres::DynamicNumericDiffCostFunction<PriorCostFunction>(
        new PriorCostFunction(prior.Wid_avg, prior.inv_sigma_Wid,
                              prior.weight_Wid * prior_scale));
    #else
    ceres::DynamicNumericDiffCostFunction<PriorCostFunction_fast> *prior_cost_function =
      new ceres::DynamicNumericDiffCostFunction<PriorCostFunction_fast>(
        new PriorCostFunction_fast(prior.Wid_avg, prior.inv_sigma_Wid_diag,
                                   prior.weight_Wid * prior_scale));
    #endif
    prior_cost_function->AddParameterBlock(params.size());
    prior_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(prior_cost_function, NULL, params.data());

    // Regularization term, minimize the norm of the weight vector
    ceres::DynamicNumericDiffCostFunction<IdentityRegularizationTerm> *reg_cost_function =
      new ceres::DynamicNumericDiffCostFunction<IdentityRegularizationTerm>(
        new IdentityRegularizationTerm(10.0)
      );
    reg_cost_function->AddParameterBlock(params.size());
    reg_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(reg_cost_function, NULL, params.data());

    // Bounds
    for(int i=0;i<params.size();++i) {
      problem.SetParameterLowerBound(params.data(), i, prior.Uid_min(i));
      problem.SetParameterUpperBound(params.data(), i, prior.Uid_max(i));
    }
  }

  // Solve it
  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Identity optimization] Problem solve time = %w seconds.\n");
    ceres::Solver::Options options;
    options.max_num_iterations = iteration;

    options.num_threads = 8;
    options.num_linear_solver_threads = 8;

    double under_relax_factor = 0.5;

#if 1
    options.initial_trust_region_radius = 1.0;
    options.min_trust_region_radius = 0.75;
    options.max_trust_region_radius = 1.25;
    options.min_lm_diagonal = 1.0;
    options.max_lm_diagonal = 1.0;
#else
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::LBFGS;
#endif

    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.FullReport())

    // Update the model parameters
    DEBUG_OUTPUT(params_model.Wid.transpose() << endl << " -> " << endl <<
                 params.transpose())
    params_model.Wid = (1.0 - under_relax_factor) * params_model.Wid + under_relax_factor * params;
    params = params_model.Wid;
  }
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::UpdateContourIndices(int iterations) {
  boost::timer::auto_cpu_timer timer(
    "[Contour update] Contour vertices update time = %w seconds.\n");
  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1],
                                 params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0],
                                              params_model.T[1],
                                              params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  //0:34
  //35:39
  //40:74

  vector<pair<int, glm::dvec4>> candidates_left;
  vector<pair<int, glm::dvec4>> candidates_center;
  vector<pair<int, glm::dvec4>> candidates_right;

  for (size_t j = 0; j < contour_indices.size(); ++j) {
    vector<double> dot_products(contour_indices[j].size(), 0.0);
    vector<glm::dvec4> contour_vertices(contour_indices[j].size());
    for (size_t i = 0; i < contour_indices[j].size(); ++i) {
//      auto model_ji = model.project(vector<int>(1, contour_indices[j][i]));
//      model_ji.ApplyWeights(params_model.Wid, params_model.Wexp);
//      auto tm = model_ji.GetTM();
//      glm::dvec4 p0(tm[0], tm[1], tm[2], 1.0);

      Vector3d v_ji = mesh.vertex(contour_indices[j][i]);
      glm::dvec4 p0(v_ji[0], v_ji[1], v_ji[2], 1.0);

      // Apply the rotation and translation as well
      glm::dvec4 p = Mview * p0;
      contour_vertices[i] = p0;

      // Compute the normal for this vertex
      auto n0 = mesh.vertex_normal(contour_indices[j][i]);
      // Normal matrix is transpose(inverse(modelView))
      glm::dvec4 n = glm::transpose(glm::inverse(Mview)) *
                     glm::dvec4(n0[0], n0[1], n0[2], 1.0);

      // Compute the dot product of normal and view direction
      glm::dvec3 view_vector(0, 0, -1);
      dot_products[i] = glm::dot(glm::normalize(glm::dvec3(n.x, n.y, n.z)),
                                 glm::normalize(view_vector));

      dot_products[i] = fabs(dot_products[i]);

      //if(n.z < 0) dot_products[i] = 1e6;
    }

    auto min_iter = std::min_element(dot_products.begin(), dot_products.end());
    int min_idx = min_iter - dot_products.begin();
    //cout << min_idx << endl;

    vector<pair<int, glm::dvec4>> *candidates;
    if (j < 35) {
      // left set
      candidates = &candidates_left;
    } else if (j >= 40) {
      // right set
      candidates = &candidates_right;
    } else {
      // center set
      candidates = &candidates_center;
    }

    candidates->push_back(make_pair(contour_indices[j][min_idx],
                                    contour_vertices[min_idx]));

#if 1
    if (min_idx > 0) {
      Vector3d v_ji1 = mesh.vertex(contour_indices[j][min_idx - 1]);
      glm::dvec4 p1(v_ji1[0], v_ji1[1], v_ji1[2], 1.0);
      candidates->push_back(make_pair(contour_indices[j][min_idx - 1],
                                      p1));
    }
    if (min_idx < static_cast<int>(contour_indices[j].size() - 1)) {
      Vector3d v_ji1 = mesh.vertex(contour_indices[j][min_idx + 1]);
      glm::dvec4 p1(v_ji1[0], v_ji1[1], v_ji1[2], 1.0);
      candidates->push_back(make_pair(contour_indices[j][min_idx + 1],
                                      p1));
    }
#endif
  }

  // Project all points to image plane and choose the closest ones as new
  // contour points.
  vector<glm::dvec3> projected_points_left(candidates_left.size());
  vector<glm::dvec3> projected_points_center(candidates_center.size());
  vector<glm::dvec3> projected_points_right(candidates_right.size());

  auto project_candidate_points = [=](
    const vector<pair<int, glm::dvec4>> &candidates,
    vector<glm::dvec3> &projected_points) {
    for (size_t i = 0; i < candidates.size(); ++i) {
      projected_points[i] = ProjectPoint(
        glm::dvec3(candidates[i].second.x,
                   candidates[i].second.y,
                   candidates[i].second.z),
        Mview, params_cam);
      //cout << projected_points[i].x << ", " << projected_points[i].y << endl;
    }
  };
  project_candidate_points(candidates_left, projected_points_left);
  project_candidate_points(candidates_center, projected_points_center);
  project_candidate_points(candidates_right, projected_points_right);

  // Find closest match for each contour point
  const int num_contour_points = 15;
  for (int i = 0; i < num_contour_points; ++i) {
    vector<pair<int, glm::dvec4>> *candidates;
    vector<glm::dvec3> *projected_points;
    if (i < 7) {
      candidates = &candidates_left;
      projected_points = &projected_points_left;
    } else if (i > 8) {
      candidates = &candidates_right;
      projected_points = &projected_points_right;
    } else {
      candidates = &candidates_center;
      projected_points = &projected_points_center;
    }

    vector<double> dists(candidates->size());
    for (size_t j = 0; j < candidates->size(); ++j) {
      double dx = (*projected_points)[j].x - params_recon.cons[i].data.x;
      double dy = (*projected_points)[j].y - params_recon.cons[i].data.y;
      dists[j] = dx * dx + dy * dy;
    }
    auto min_iter = std::min_element(dists.begin(), dists.end());
    double min_acceptable_dist = 100 * iterations * iterations;
    if (sqrt(*min_iter) > min_acceptable_dist) {
      //cout << sqrt(*min_iter) << endl;
      continue;
    } else {
      //cout << i << ": " << indices[i] << " -> " << candidates[min_iter - dists.begin()].first << endl;
      indices[i] = (*candidates)[min_iter - dists.begin()].first;
      params_recon.cons[i].vidx = (*candidates)[min_iter - dists.begin()].first;
      params_model.vindices(i) = params_recon.cons[i].vidx;
      model_projected[i] = model.project(vector<int>(1, indices[i]));
      model_projected[i].ApplyWeights(params_model.Wid, params_model.Wexp);
    }
  }
}

#endif // MULTILINEARRECONSTRUCTOR_HPP
