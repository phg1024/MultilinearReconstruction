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

  friend istream& operator>>(istream& is, CameraParameters& params);
  friend ostream& operator<<(ostream& os, const CameraParameters& params);

  double fovy;
  double far;
  double focal_length;
  glm::dvec2 image_plane_center;
  glm::dvec2 image_size;
};

inline istream& operator>>(istream& is, CameraParameters& params) {
  is >> params.fovy >> params.far >> params.focal_length
     >> params.image_plane_center.x >> params.image_plane_center.y
     >> params.image_size.x >> params.image_size.y;
  return is;
}

inline ostream& operator<<(ostream& os, const CameraParameters& params) {
  os << params.fovy << endl;
  os << params.far << endl;
  os << params.focal_length << endl;
  os << params.image_plane_center.x << ' ' << params.image_plane_center.y << endl;
  os << params.image_size.x << ' ' << params.image_size.y;
  return os;
}

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

  friend istream& operator>>(istream& is, ModelParameters& params);
  friend ostream& operator<<(ostream& os, const ModelParameters& params);

  static const int nFACSDim = 47;
  VectorXd Wid;               // identity weights
  VectorXd Wexp, Wexp_FACS;   // expression weights
  Vector3d R;              // rotation
  Vector3d T;                 // translation
};

namespace {
  void write_vector(ostream& os, const VectorXd& v) {
    os << v.rows() << ' ';
    for(int i=0;i<v.rows();++i) {
      os << v(i) << ' ';
    }
    os << endl;
  }
  void read_vector(istream& is, VectorXd& v) {
    int nrows;
    is >> nrows;
    v.resize(nrows, 1);
    for(int i=0;i<nrows;++i) is >> v(i);
  }
}

inline istream& operator>>(istream& is, ModelParameters& params) {
  read_vector(is, params.Wid);
  read_vector(is, params.Wexp);
  read_vector(is, params.Wexp_FACS);
  is >> params.R(0) >> params.R(1) >> params.R(2);
  is >> params.T(0) >> params.T(1) >> params.T(2);
  return is;
}

inline ostream& operator<<(ostream& os, const ModelParameters& params) {
  write_vector(os, params.Wid);
  write_vector(os, params.Wexp);
  write_vector(os, params.Wexp_FACS);
  os << params.R(0) << ' ' << params.R(1) << ' ' << params.R(2) << endl;
  os << params.T(0) << ' ' << params.T(1) << ' ' << params.T(2);
  return os;
}

struct ReconstructionResult {
  CameraParameters params_cam;
  ModelParameters params_model;
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
