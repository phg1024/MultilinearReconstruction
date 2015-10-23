#ifndef MULTILINEARRECONSTRUCTION_PARAMETERS_H
#define MULTILINEARRECONSTRUCTION_PARAMETERS_H

#include "mathutils.hpp"

struct CameraParameters {
  CameraParameters() {}
  CameraParameters(double fovy, double far, int image_width, int image_height)
    : fovy(fovy), far(far) {
    focal_length = 1.0 / tan(0.5 * fovy);
    image_plane_center = glm::vec2(image_width * 0.5, image_height * 0.5);
    image_size = glm::vec2(image_width, image_height);
  }

  static CameraParameters DefaultParameters(int image_width,
                                            int image_height) {
    const double fovy = deg2rad(40.0);
    const double far = 100.0;
    return CameraParameters(fovy, far, image_width, image_height);
  }

  double fovy;
  double far;
  double focal_length;
  glm::dvec2 image_plane_center;
  glm::dvec2 image_size;
};

struct ModelParameters {
  static ModelParameters DefaultParameters(const MatrixXd& Uid,
                                           const MatrixXd& Uexp) {
    ModelParameters model_params;

    // Make a neutral face
    model_params.Wexp_FACS.resize(ModelParameters::nFACSDim);
    model_params.Wexp_FACS(0) = 1.0;
    for (int i = 1; i < ModelParameters::nFACSDim; ++i)
      model_params.Wexp_FACS(i) = 0.0;
    model_params.Wexp = model_params.Wexp_FACS.transpose() * Uexp;

    // Use average identity
    model_params.Wid = Uid.row(0);

    model_params.R = Vector3d(0, 0, 0);
    model_params.T = Vector3d(0, 0, -1.0);
    
    return model_params;
  }

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

struct OptimizationParameters {
  int maxIters;
  double errorThreshold;
  double errorDiffThreshold;
};



#endif //MULTILINEARRECONSTRUCTION_PARAMETERS_H
