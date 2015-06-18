#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include "common.h"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

template <typename T>
struct Constraint {
  int vidx;         // vertex index
  double weight;    // weight for this constraint
  T data;
};

using Constraint2D = Constraint<Vector2d>;
using Constraint3D = Constraint<Vector3d>;
using Constraint2D_Depth = Constraint<Vector3d>;

#endif // CONSTRAINTS_H

