#ifndef MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H
#define MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

#include "ceres/ceres.h"

#include "basicmesh.h"
#include "common.h"
#include "constraints.h"
#include "costfunctions.h"
#include "multilinearmodel.h"
#include "parameters.h"
#include "singleimagereconstructor.hpp"
#include "utils.hpp"

using namespace Eigen;

template <typename Constraint>
class MultiImageReconstructor {
public:
  MultiImageReconstructor() {}

  void LoadModel(const string& filename) {
    model = MultilinearModel(filename);
  }
  void LoadPrior(const string& filename_id, const string& filename_exp) {
    prior.load(filename_id, filename_exp);
  }
  void SetContourIndices(const vector<vector<int>>& contour_indices_in) {
    contour_indices = contour_indices_in;
  }
  void SetMesh(const BasicMesh& mesh) {
    template_mesh = mesh;
  }
  void SetIndices(const vector<int>& indices) {
    init_indices = indices;
  }

  void AddImagePointsPair(const pair<QImage, vector<Constraint>>& p) {
    image_points_pairs.push_back(p);
  }

  bool Reconstruct();

  const Vector3d& GetRotation(int imgidx) const { return param_sets[imgidx].model.R; }
  const Vector3d& GetTranslation(int imgidx) const { return param_sets[imgidx].model.T; }
  const VectorXd& GetIdentityWeights(int imgidx) const { return param_sets[imgidx].model.Wid; }
  const VectorXd& GetExpressionWeights(int imgidx) const { return param_sets[imgidx].model.Wexp_FACS; }
  const Tensor1& GetGeometry(int imgidx) const {
    model.ApplyWeights(GetIdentityWeights(imgidx), GetExpressionWeights(imgidx));
    return model.GetTM();
  }
  const CameraParameters GetCameraParameters(int imgidx) const { return param_sets[imgidx].cam; }
  const vector<int> GetIndices(int imgidx) const { return param_sets[imgidx].indices; }
  vector<int> GetUpdatedIndices(int imgidx) const {
    vector<int> idxs;
    for(int i=0;i<param_sets[imgidx].recon.cons.size();++i) {
      idxs.push_back(param_sets[imgidx].recon.cons[i].vidx);
    }
    return idxs;
  }

protected:

private:
  MultilinearModel model;
  MultilinearModelPrior prior;
  vector<vector<int>> contour_indices;
  vector<int> init_indices;
  BasicMesh template_mesh;

  struct ParameterSet {
    vector<int> indices;
    BasicMesh mesh;

    CameraParameters cam;
    ModelParameters model;
    ReconstructionParameters<Constraint> recon;
    OptimizationParameters opt;
  };

  // Input image points pairs
  vector<pair<QImage, vector<Constraint>>> image_points_pairs;

  // A set of parameters for each image
  vector<ParameterSet> param_sets;
};

template <typename Constraint>
bool MultiImageReconstructor<Constraint>::Reconstruct() {
  // @todo Work on this function.

  // Initialize the parameter sets
  for(auto& params : param_sets) {
    params.indices = init_indices;
    params.mesh = template_mesh;

    // camera parameters

    // model parameters

  }

  // Main reconstruction loop
  //  1. Use single image reconstructor to do per-image reconstruction first
  //  2. Select a consistent set of images for joint reconstruction
  //  3. Convergence test. If not converged, goto step 1.
}

#endif //MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H
