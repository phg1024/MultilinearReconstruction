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
#include "utils.hpp"

#include "boost/timer/timer.hpp"

#define USE_ANALYTIC_COST_FUNCTIONS 1

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
    : opt_mode(All), need_precise_result(false), is_parameters_initialized(false) {}

  void LoadModel(const string &filename) { model = MultilinearModel(filename); }

  void LoadPriors(const string &filename_id, const string &filename_exp) {
    prior.load(filename_id, filename_exp);
  }

  void SetContourIndices(
    const vector<vector<int>> &contour_points) { contour_indices = contour_points; }

  void SetConstraints(
    const vector<Constraint> &cons) { params_recon.cons = cons; }

  void SetImageSize(int w, int h) {
    params_recon.imageWidth = w;
    params_recon.imageHeight = h;
  }

  void SetMesh(const BasicMesh &mesh_in) {
    mesh = mesh_in;
  }

  void SetOptimizationParameters(const OptimizationParameters &params) {
    params_opt = params;
  }

  void SetInitialParameters(const ModelParameters& model_params,
                            const CameraParameters& camera_params);

  void SetOptimizationMode(OptimizationMode mode) {
    opt_mode = mode;
  }

  bool Reconstruct();

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

  vector<int> GetUpdatedIndices() const {
    vector<int> idxs;
    for (int i = 0; i < params_recon.cons.size(); ++i) {
      idxs.push_back(params_recon.cons[i].vidx);
    }
    return idxs;
  }

  void SaveReconstructionResults(const string& filename) const {
    ofstream fout(filename);
    fout << params_cam << endl;
    fout << params_model << endl;
    fout.close();
  }

protected:
  void InitializeParameters();

  void UpdateModels();

  void OptimizeForPosition();

  void OptimizeForPose(int max_iterations);

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

  CameraParameters params_cam;
  ModelParameters params_model;
  ReconstructionParameters<Constraint> params_recon;
  OptimizationParameters params_opt;

  OptimizationMode opt_mode;

  bool need_precise_result;
  bool is_parameters_initialized;
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
void SingleImageReconstructor<Constraint>::InitializeParameters() {
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

  // No rotation and translation
  model_params.R = Vector3d(1e-3, 1e-3, 1e-3);
  model_params.T = Vector3d(0, 0, -1.0);

  SetInitialParameters(model_params, camera_params);
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::UpdateModels() {
  model.ApplyWeights(params_model.Wid, params_model.Wexp);

  for (int i = 0; i < indices.size(); ++i) {
    params_recon.cons[i].vidx = indices[i];
    params_recon.cons[i].weight = 1.0;
  }

  // Create initial projected models
  model_projected.resize(params_recon.cons.size());
  for (int i = 0; i < params_recon.cons.size(); ++i) {
    model_projected[i] = model.project(vector<int>(1, indices[i]));
    model_projected[i].ApplyWeights(params_model.Wid, params_model.Wexp);
  }
}

template<typename Constraint>
bool SingleImageReconstructor<Constraint>::Reconstruct() {
  // Initialize parameters
  cout << "Reconstruction begins." << endl;

  const int num_contour_points = 15;

  // Reconstruction begins
  if (!is_parameters_initialized) {
    InitializeParameters();
  }

  for (int i = 0; i < num_contour_points; ++i) {
    params_recon.cons[i].weight = 10.0;
  }

  ColorStream(ColorOutput::Red) << "Initial Error = " << ComputeError();

  // Optimization parameters
  const int kMaxIterations = need_precise_result ? 8 : 4;
  const double init_weights = 1.0;
  prior.weight_Wid = 100.0;
  const double d_wid = 20.0;
  prior.weight_Wexp = 10.0;
  const double d_wexp = 2.0;
  int iters = 0;

  // Before entering the main loop, estimate the translation first
  OptimizeForPosition();

  while (iters++ < kMaxIterations) {
    ColorStream(ColorOutput::Green) << "Iteration " << iters << " begins.";
    {
      boost::timer::auto_cpu_timer timer_loop(
        "[Main loop] Iteration time = %w seconds.\n");

      if((opt_mode & (Identity | Expression))){
        boost::timer::auto_cpu_timer timer(
          "[Main loop] Multilinear model weights update time = %w seconds.\n");
        model.ApplyWeights(params_model.Wid, params_model.Wexp);
      }
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();

      if(opt_mode & Pose) {
        for (int pose_opt_iter = 0; pose_opt_iter < 2; ++pose_opt_iter) {
          OptimizeForPose(2);
          UpdateContourIndices(iters);
        }
      }

      if(opt_mode & Expression) {
        //OptimizeForExpression(iters*100);
        OptimizeForExpression_FACS(iters);
      }

      if(opt_mode & FocalLength) {
        OptimizeForFocalLength();
      }

      if(opt_mode & Expression){
        boost::timer::auto_cpu_timer timer(
          "[Main loop] Multilinear model weights update time = %w seconds.\n");
        model.ApplyWeights(params_model.Wid, params_model.Wexp);
      }
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();

      if(opt_mode & Pose) {
        for (int pose_opt_iter = 0; pose_opt_iter < 2; ++pose_opt_iter) {
          OptimizeForPose(2);
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
      prior.weight_Wid -= d_wid;
      prior.weight_Wexp -= d_wexp;
      for (int i = 0; i < num_contour_points; ++i) {
        params_recon.cons[i].weight = sqrt(params_recon.cons[i].weight);
      }
    }
    ColorStream(ColorOutput::Green) << "Iteration " << iters << " finished.";
  }

  cout << "Reconstruction done." << endl;
  model.ApplyWeights(params_model.Wid, params_model.Wexp);

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

  double E = 0;
  for (int i = 0; i < indices.size(); ++i) {
    auto &model_i = model_projected[i];
    //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
    auto tm = model_i.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    auto q = ProjectPoint(p, Mview, params_cam);
    double dx = q.x - params_recon.cons[i].data.x;
    double dy = q.y - params_recon.cons[i].data.y;
    E += dx * dx + dy * dy;
  }

  return E;
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

    for (int i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
#if USE_ANALYTIC_COST_FUNCTIONS
      ceres::CostFunction *cost_function = new PositionCostFunction_analytic(
        model_i,
        params_recon.cons[i],
        params_cam);
#else
      ceres::CostFunction *cost_function =
        new ceres::NumericDiffCostFunction<PositionCostFunction, ceres::CENTRAL, 1, 3>(
          new PositionCostFunction(model_i,
                                   params_recon.cons[i],
                                   params_cam));
#endif
      problem.AddResidualBlock(cost_function, NULL, params.data());
    }
  }

  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Position optimization] Problem solve time = %w seconds.\n");

    ceres::Solver::Options options;
    options.max_num_iterations = 30;
    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport());
  }

  Vector3d newT(params[0], params[1], params[2]);

  DEBUG_OUTPUT(
    "T: " << params_model.T.transpose() << " -> " << newT.transpose());

  params_model.T = newT;
}

template<typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForPose(int max_iters) {
  boost::timer::auto_cpu_timer timer_all(
    "[Pose optimization] Total time = %w seconds.\n");

  ceres::Problem problem;
  vector<double> params{params_model.R[0], params_model.R[1], params_model.R[2],
                        params_model.T[0], params_model.T[1],
                        params_model.T[2]};

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Pose optimization] Problem construction time = %w seconds.\n");
    for (int i = 0; i < indices.size(); ++i) {
      auto &model_i = model_projected[i];
      //model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
#if USE_ANALYTIC_COST_FUNCTIONS
      ceres::CostFunction *cost_function =
        new PoseCostFunction_analytic(model_i, params_recon.cons[i],
                                      params_cam);
      problem.AddResidualBlock(cost_function, NULL, params.data(),
                               params.data() + 3);
#else
      ceres::CostFunction *cost_function =
        new ceres::NumericDiffCostFunction<PoseCostFunction, ceres::CENTRAL, 1, 6>(
          new PoseCostFunction(model_i,
                               params_recon.cons[i],
                               params_cam));
      problem.AddResidualBlock(cost_function, NULL, params.data());
#endif
    }
  }

  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Pose optimization] Problem solve time = %w seconds.\n");

    ceres::Solver::Options options;
    options.max_num_iterations = max_iters;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::LBFGS;
    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;

    Solve(options, &problem, &summary);
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
  for (int i = 0; i < indices.size(); ++i) {
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
  double prior_scale = 100.0 / puple_distance;

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
    Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport())

    options.max_num_iterations = iteration * 5;
    options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    Solve(options, &problem, &summary);
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
  double prior_scale = 100.0 / puple_distance;

  // Define the optimization problem
  ceres::Problem problem;

  VectorXd params = params_model.Wexp_FACS;

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Expression optimization] Problem construction time = %w seconds.\n");
    for (int i = 0; i < indices.size(); ++i) {
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

    ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationCostFunction> *prior_cost_function =
      new ceres::DynamicNumericDiffCostFunction<ExpressionRegularizationCostFunction>(
        new ExpressionRegularizationCostFunction(prior.Wexp_avg,
                                                 prior.inv_sigma_Wexp,
                                                 prior.Uexp, prior.weight_Wexp *
                                                             prior_scale));
    prior_cost_function->AddParameterBlock(params.size()-1);
    prior_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(prior_cost_function, NULL, params.data()+1);
  }

  // Solve it
  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Expression optimization] Problem solve time = %w seconds.\n");
    ceres::Solver::Options options;
    options.max_num_iterations = iteration * 3;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.BriefReport())

    if (need_precise_result) {
      options.max_num_iterations = iteration * 5;
      options.line_search_direction_type = ceres::STEEPEST_DESCENT;
      Solve(options, &problem, &summary);
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
  double prior_scale = 100.0 / puple_distance;

  // Define the optimization problem
  ceres::Problem problem;
  VectorXd params = params_model.Wid;

  {
    boost::timer::auto_cpu_timer timer_construction(
      "[Identity optimization] Problem construction time = %w seconds.\n");
    for (int i = 0; i < indices.size(); ++i) {
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

    ceres::DynamicNumericDiffCostFunction<PriorCostFunction> *prior_cost_function =
      new ceres::DynamicNumericDiffCostFunction<PriorCostFunction>(
        new PriorCostFunction(prior.Wid_avg, prior.inv_sigma_Wid,
                              prior.weight_Wid * prior_scale));
    prior_cost_function->AddParameterBlock(params.size());
    prior_cost_function->SetNumResiduals(1);
    problem.AddResidualBlock(prior_cost_function, NULL, params.data());
  }

  // Solve it
  {
    boost::timer::auto_cpu_timer timer_solve(
      "[Identity optimization] Problem solve time = %w seconds.\n");
    ceres::Solver::Options options;
    options.max_num_iterations = iteration;
    //options.minimizer_type = ceres::LINE_SEARCH;
    //options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
    DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    DEBUG_OUTPUT(summary.FullReport())

    if (need_precise_result) {
      options.max_num_iterations = iteration * 5;
      options.line_search_direction_type = ceres::STEEPEST_DESCENT;
      Solve(options, &problem, &summary);
      DEBUG_OUTPUT(summary.FullReport())
    }
  }

  // Update the model parameters
  DEBUG_OUTPUT(params_model.Wid.transpose() << endl << " -> " << endl <<
               params.transpose())
  params_model.Wid = params;
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

  for (int j = 0; j < contour_indices.size(); ++j) {
    vector<double> dot_products(contour_indices[j].size(), 0.0);
    vector<glm::dvec4> contour_vertices(contour_indices[j].size());
    for (int i = 0; i < contour_indices[j].size(); ++i) {
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
                                 view_vector);

      dot_products[i] = fabs(dot_products[i]);

      if(n.z < 0) dot_products[i] = 1e6;
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
    if (min_idx < contour_indices[j].size() - 1) {
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
    for (int i = 0; i < candidates.size(); ++i) {
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
    for (int j = 0; j < candidates->size(); ++j) {
      double dx = (*projected_points)[j].x - params_recon.cons[i].data.x;
      double dy = (*projected_points)[j].y - params_recon.cons[i].data.y;
      dists[j] = dx * dx + dy * dy;
    }
    auto min_iter = std::min_element(dists.begin(), dists.end());
    double min_acceptable_dist = 5 * iterations * iterations;
    if (sqrt(*min_iter) > min_acceptable_dist) {
      //cout << sqrt(*min_iter) << endl;
      continue;
    } else {
      //cout << i << ": " << indices[i] << " -> " << candidates[min_iter - dists.begin()].first << endl;
      indices[i] = (*candidates)[min_iter - dists.begin()].first;
      params_recon.cons[i].vidx = (*candidates)[min_iter - dists.begin()].first;
      model_projected[i] = model.project(vector<int>(1, indices[i]));
      model_projected[i].ApplyWeights(params_model.Wid, params_model.Wexp);
    }
  }
}

#endif // MULTILINEARRECONSTRUCTOR_HPP
