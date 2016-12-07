#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include "common.h"

#define GLM_ENFORCE_SWIZZLE
//#include "glm/glm.hpp"
#include "glm/vector_relational.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

template <typename T>
struct Constraint {
  Constraint() : vidx(-1), weight(1.0), data(T()) {}

  int vidx;         // vertex index
  double weight;    // weight for this constraint
  T data;
};

using Constraint2D = Constraint<glm::dvec2>;
using Constraint3D = Constraint<glm::dvec3>;
using Constraint2D_Depth = Constraint<glm::dvec3>;

#endif // CONSTRAINTS_H
