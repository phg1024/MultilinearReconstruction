#ifndef MULTILINEARRECONSTRUCTOR_HPP
#define MULTILINEARRECONSTRUCTOR_HPP

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

#include "ceres/ceres.h"

#include "common.h"
#include "constraints.h"
#include "costfunctions.h"
#include "multilinearmodel.h"
#include "parameters.h"
#include "utils.hpp"

using namespace Eigen;

struct ModelParameters {
  static const int nFACSDim = 47;
  VectorXd Wid;               // identity weights
  VectorXd Wexp, Wexp_FACS;   // expression weights
  Vector3d R;              // rotation
  Vector3d T;                 // translation
};

template <typename Constraint>
struct ReconstructionParameters {
  int imageWidth, imageHeight;
  vector<Constraint> cons;
};

struct MultilinearModelPrior {
  VectorXd Wid_avg, Wexp_avg;
  VectorXd Wid0, Wexp0;       // identity and expression prior
  MatrixXd Uid, Uexp;
  MatrixXd sigma_Wid, sigma_Wexp;
  MatrixXd inv_sigma_Wid, inv_sigma_Wexp;
  double weight_Wid, weight_Wexp;

  void load(const string &filename_id, const string &filename_exp) {
    cout << "loading prior data ..." << endl;
    const string fnwid = filename_id;
    ifstream fwid(fnwid, ios::in | ios::binary);

    int ndims;
    fwid.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    cout << "identity prior dim = " << ndims << endl;

    Wid_avg.resize(ndims);
    Wid0.resize(ndims);
    sigma_Wid.resize(ndims, ndims);

    fwid.read(reinterpret_cast<char*>(Wid_avg.data()), sizeof(double)*ndims);
    fwid.read(reinterpret_cast<char*>(Wid0.data()), sizeof(double)*ndims);
    fwid.read(reinterpret_cast<char*>(sigma_Wid.data()), sizeof(double)*ndims*ndims);

    int m, n;
    fwid.read(reinterpret_cast<char*>(&m), sizeof(int));
    fwid.read(reinterpret_cast<char*>(&n), sizeof(int));
    cout << "Uid size: " << m << 'x' << n << endl;
    Uid.resize(m, n);
    fwid.read(reinterpret_cast<char*>(Uid.data()), sizeof(double)*m*n);

    fwid.close();

    message("identity prior loaded.");
    /*
    cout << "Wid_avg = " << Wid_avg << endl;
    cout << "Wid0 = " << Wid0 << endl;
    cout << "sigma_Wid = " << sigma_Wid << endl;
    cout << "Uid = " << Uid << endl;
    */

    message("processing identity prior.");
    inv_sigma_Wid = sigma_Wid.inverse();
    message("done");

    const string fnwexp = filename_exp;
    ifstream fwexp(fnwexp, ios::in | ios::binary);

    fwexp.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    cout << "expression prior dim = " << ndims << endl;

    Wexp0.resize(ndims);
    Wexp_avg.resize(ndims);
    sigma_Wexp.resize(ndims, ndims);

    fwexp.read(reinterpret_cast<char*>(Wexp_avg.data()), sizeof(double)*ndims);
    fwexp.read(reinterpret_cast<char*>(Wexp0.data()), sizeof(double)*ndims);
    fwexp.read(reinterpret_cast<char*>(sigma_Wexp.data()), sizeof(double)*ndims*ndims);

    fwexp.read(reinterpret_cast<char*>(&m), sizeof(int));
    fwexp.read(reinterpret_cast<char*>(&n), sizeof(int));
    cout << "Uexp size: " << m << 'x' << n << endl;
    Uexp.resize(m, n);
    fwexp.read(reinterpret_cast<char*>(Uexp.data()), sizeof(double)*m*n);

    fwexp.close();

    message("expression prior loaded.");
    /*
    cout << "Wexp_avg = " << Wexp_avg << endl;
    cout << "Wexp0 = " << Wexp0 << endl;
    cout << "sigma_Wexp = " << sigma_Wexp << endl;
    cout << "Uexp = " << Uexp << endl;
    */
    message("processing expression prior.");
    inv_sigma_Wexp = sigma_Wexp.inverse();
    message("done.");
  }
};

struct OptimizationParameters {
  int maxIters;
  double errorThreshold;
  double errorDiffThreshold;
};

template <typename Constraint>
class SingleImageReconstructor {
public:
  SingleImageReconstructor(){}
  void LoadModel(const string &filename);
  void LoadPriors(const string &filename_id, const string &filename_exp);
  void SetIndices(const vector<int> &indices_vec) { indices = indices_vec; }

  void SetConstraints(const vector<Constraint> &cons) { params_recon.cons = cons; }
  void SetImageSize(int w, int h) {
    params_recon.imageWidth = w;
    params_recon.imageHeight = h;
  }
  void SetOptimizationParameters(const OptimizationParameters &params) {
    params_opt = params;
  }

  bool Reconstruct();

  const Vector3d& GetRotation() const { return params_model.R; }
  const Vector3d& GetTranslation() const { return params_model.T; }
  const VectorXd& GetIdentityWeights() const { return params_model.Wid; }
  const VectorXd& GetExpressionWeights() const { return params_model.Wexp_FACS; }
  const Tensor1& GetGeometry() const { return model.GetTM(); }
  const CameraParameters GetCameraParameters() const { return params_cam; }

protected:
  void OptimizeForPose();
  void OptimizeForExpression();
  void OptimizeForIdentity();

private:
  MultilinearModel model, model_projected;
  vector<int> indices;
  MultilinearModelPrior prior;

  CameraParameters params_cam;
  ModelParameters params_model;
  ReconstructionParameters<Constraint> params_recon;
  OptimizationParameters params_opt;
};

template <typename Constraint>
void SingleImageReconstructor<Constraint>::LoadModel(const string &filename)
{
  model = MultilinearModel(filename);
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::LoadPriors(const string &filename_id, const string &filename_exp)
{
  prior.load(filename_id, filename_exp);
}

template <typename Constraint>
bool SingleImageReconstructor<Constraint>::Reconstruct()
{
  // Initialize parameters
  cout << "Reconstruction begins." << endl;

  // Camera parameters
  params_cam.focal_length = glm::vec2(1000.0, 1000.0);
  params_cam.image_plane_center = glm::vec2(params_recon.imageWidth * 0.5,
                                            params_recon.imageHeight * 0.5);
  params_cam.image_size = glm::vec2(params_recon.imageWidth,
                                    params_recon.imageHeight);

  // Model parameters

  // Make a neutral face
  params_model.Wexp_FACS.resize(ModelParameters::nFACSDim);
  params_model.Wexp_FACS(0) = 1.0;
  for(int i=1;i<ModelParameters::nFACSDim;++i) params_model.Wexp_FACS(i) = 0.0;
  params_model.Wexp = params_model.Wexp_FACS.transpose() * prior.Uexp;

  // Use average identity
  params_model.Wid = prior.Wid_avg;

  // No rotation and translation
  params_model.R = Vector3d(0, 0, 0);
  params_model.T = Vector3d(0, 0, -5.0);

  model.ApplyWeights(params_model.Wid, params_model.Wexp);

  // Reconstruction begins
  const int kMaxIterations = 8;

  int iters = 0;
  while( iters++ < kMaxIterations ) {
    OptimizeForPose();
    OptimizeForExpression();
    OptimizeForIdentity();
  }

  cout << "Reconstruction done." << endl;
  model.ApplyWeights(params_model.Wid, params_model.Wexp);

  return true;
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForPose() {
  ceres::Problem problem;
  vector<double> params{params_model.R[0], params_model.R[1], params_model.R[2],
                        params_model.T[0], params_model.T[1], params_model.T[2]};

  for(int i=0;i<indices.size();++i) {
    auto model_i = model.project(vector<int>(1, indices[i]));
    model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
    ceres::CostFunction *cost_function =
      new ceres::NumericDiffCostFunction<PoseCostFunction, ceres::CENTRAL, 2, 6>(
        new PoseCostFunction(model_i,
                             params_recon.cons[i],
                             params_cam));
    problem.AddResidualBlock(cost_function, NULL, params.data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;
  Vector3d newR(params[0], params[1], params[2]);
  Vector3d newT(params[3], params[4], params[5]);
  cout << "R: " << params_model.R.transpose() << " -> " << newR.transpose() << endl;
  cout << "T: " << params_model.T.transpose() << " -> " << newT.transpose() << endl;
  params_model.R = newR;
  params_model.T = newT;
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForExpression() {
  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1], params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0], params_model.T[1], params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  VectorXd params = params_model.Wexp_FACS;

  // Define the optimization problem
  ceres::Problem problem;

  for(int i=0;i<indices.size();++i) {
    auto model_i = model.project(vector<int>(1, indices[i]));
    model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
    ceres::DynamicNumericDiffCostFunction<ExpressionCostFunction> *cost_function =
      new ceres::DynamicNumericDiffCostFunction<ExpressionCostFunction>(
        new ExpressionCostFunction(model_i,
                                   params_recon.cons[i],
                                   params.size(),
                                   Mview,
                                   prior.Uexp,
                                   params_cam));
    cost_function->AddParameterBlock(params.size());
    cost_function->SetNumResiduals(2);
    problem.AddResidualBlock(cost_function, NULL, params.data());
  }

  for(int i=0;i<params.size();++i) {
    problem.SetParameterLowerBound(params.data(), i, 0.0);
    problem.SetParameterUpperBound(params.data(), i, 1.0);
  }

  // Solve it
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;

  // Update the model parameters
  cout << params_model.Wexp_FACS.transpose() << endl
       << " -> " << endl
       << params.transpose() << endl;
  params_model.Wexp_FACS = params;
  params_model.Wexp = params_model.Wexp_FACS.transpose() * prior.Uexp;
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::OptimizeForIdentity() {
  // Create view matrix
  auto Rmat = glm::eulerAngleYXZ(params_model.R[0], params_model.R[1], params_model.R[2]);
  glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                   glm::dvec3(params_model.T[0], params_model.T[1], params_model.T[2]));
  glm::dmat4 Mview = Tmat * Rmat;

  VectorXd params = params_model.Wid;

  // Define the optimization problem
  ceres::Problem problem;

  for(int i=0;i<indices.size();++i) {
    auto model_i = model.project(vector<int>(1, indices[i]));
    model_i.ApplyWeights(params_model.Wid, params_model.Wexp);
    ceres::DynamicNumericDiffCostFunction<IdentityCostFunction> *cost_function =
      new ceres::DynamicNumericDiffCostFunction<IdentityCostFunction>(
        new IdentityCostFunction(model_i,
                                   params_recon.cons[i],
                                   params.size(),
                                   Mview,
                                   params_cam));
    cost_function->AddParameterBlock(params.size());
    cost_function->SetNumResiduals(2);
    problem.AddResidualBlock(cost_function, NULL, params.data());
  }

  // Solve it
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  cout << summary.BriefReport() << endl;

  // Update the model parameters
  cout << params_model.Wid.transpose() << endl
  << " -> " << endl
  << params.transpose() << endl;
  params_model.Wid = params;
}

#endif // MULTILINEARRECONSTRUCTOR_HPP

