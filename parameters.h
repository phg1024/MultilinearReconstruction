#ifndef MULTILINEARRECONSTRUCTION_PARAMETERS_H
#define MULTILINEARRECONSTRUCTION_PARAMETERS_H

struct CameraParameters {
  double fovy;
  double far;
  double focal_length;
  glm::dvec2 image_plane_center;
  glm::dvec2 image_size;
};

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

struct OptimizationParameters {
  int maxIters;
  double errorThreshold;
  double errorDiffThreshold;
};



#endif //MULTILINEARRECONSTRUCTION_PARAMETERS_H
