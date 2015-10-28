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
#include "statsutils.h"
#include "utils.hpp"

using namespace Eigen;

template <typename Constraint>
class MultiImageReconstructor {
public:
  MultiImageReconstructor() {}

  void LoadModel(const string& filename) {
    model = MultilinearModel(filename);
    single_recon.LoadModel(filename);
  }
  void LoadPriors(const string& filename_id, const string& filename_exp) {
    prior.load(filename_id, filename_exp);
    single_recon.LoadPriors(filename_id, filename_exp);
  }
  void SetContourIndices(const vector<vector<int>>& contour_indices_in) {
    contour_indices = contour_indices_in;
    single_recon.SetContourIndices(contour_indices_in);
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

  // The worker for single image reconstruction
  SingleImageReconstructor<Constraint> single_recon;
};

template <typename Constraint>
bool MultiImageReconstructor<Constraint>::Reconstruct() {
  // TODO Work on this function.

  // Initialize the parameter sets
  param_sets.resize(image_points_pairs.size());
  for(size_t i=0;i<param_sets.size();++i) {
    auto& params = param_sets[i];
    params.indices = init_indices;
    params.mesh = template_mesh;

    const int image_width = image_points_pairs[i].first.width();
    const int image_height = image_points_pairs[i].first.height();

    // camera parameters
    cout << image_width << "x" << image_height << endl;
    params.cam = CameraParameters::DefaultParameters(image_width, image_height);
    cout << params.cam.image_size.x << ", " << params.cam.image_size.y << endl;

    // model parameters
    params.model = ModelParameters::DefaultParameters(prior.Uid, prior.Uexp);

    // reconstruction parameters
    params.recon.cons = image_points_pairs[i].second;
    params.recon.imageWidth = image_width;
    params.recon.imageHeight = image_height;
  }

  const size_t num_images = image_points_pairs.size();

  // Main reconstruction loop
  //  1. Use single image reconstructor to do per-image reconstruction first
  //  2. Select a consistent set of images for joint reconstruction
  //  3. Convergence test. If not converged, goto step 1.



  {
    // Single image reconstruction step
    for(size_t i=0;i<num_images;++i) {
      single_recon.SetMesh(param_sets[i].mesh);
      single_recon.SetIndices(param_sets[i].indices);
      single_recon.SetImageSize(param_sets[i].recon.imageWidth, param_sets[i].recon.imageHeight);
      single_recon.SetConstraints(param_sets[i].recon.cons);

      single_recon.SetInitialParameters(param_sets[i].model, param_sets[i].cam);

      {
        boost::timer::auto_cpu_timer t("Single image reconstruction finished in %w seconds.\n");
        single_recon.Reconstruct();
      }

      // Store results
      auto tm = single_recon.GetGeometry();
      param_sets[i].mesh.UpdateVertices(tm);
      param_sets[i].model = single_recon.GetModelParameters();
      param_sets[i].indices = single_recon.GetIndices();
      param_sets[i].cam = single_recon.GetCameraParameters();

      if (false) {
        // Visualize the reconstruction results
        MeshVisualizer* w = new MeshVisualizer("reconstruction result", param_sets[i].mesh);
        w->BindConstraints(image_points_pairs[i].second);
        w->BindImage(image_points_pairs[i].first);
        w->BindLandmarks(init_indices);

        w->BindUpdatedLandmarks(param_sets[i].indices);
        w->SetMeshRotationTranslation(param_sets[i].model.R, param_sets[i].model.T);
        w->SetCameraParameters(param_sets[i].cam);
        w->resize(image_points_pairs[i].first.width(), image_points_pairs[i].first.height());
        w->show();
      }
    }

    // TODO Parameters estimation step, choose a consistent set of images for joint
    // optimization
    MatrixXd identity_weights(param_sets[0].model.Wid.rows(), num_images);
    for(int i=0;i<num_images;++i) {
      identity_weights.col(i) = param_sets[i].model.Wid;
    }

    // Remove outliers
    vector<int> consistent_set = StatsUtils::FindConsistentSet(identity_weights, 0.5);

    // TODO Joint reconstruction step, obtain refined identity weights
  } // end of main reconstruction loop



  // Visualize the final reconstruction results
  for(size_t i=0;i<num_images;++i) {
    // Visualize the reconstruction results
    MeshVisualizer* w = new MeshVisualizer("reconstruction result", param_sets[i].mesh);
    w->BindConstraints(image_points_pairs[i].second);
    w->BindImage(image_points_pairs[i].first);
    w->BindLandmarks(init_indices);

    w->BindUpdatedLandmarks(param_sets[i].indices);
    w->SetMeshRotationTranslation(param_sets[i].model.R, param_sets[i].model.T);
    w->SetCameraParameters(param_sets[i].cam);
    w->resize(image_points_pairs[i].first.width(), image_points_pairs[i].first.height());
    w->show();
  }
}

#endif //MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H
