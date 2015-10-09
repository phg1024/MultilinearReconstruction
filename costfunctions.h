#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include "common.h"
#include "constraints.h"
#include "multilinearmodel.h"
#include "parameters.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/euler_angles.hpp"
#include <eigen3/Eigen/Dense>
using namespace Eigen;

inline glm::dvec3 ProjectPoint(const glm::dvec3& p, const glm::dmat4& Mview, const CameraParameters& cam_params) {
  glm::dmat4 Mproj = glm::perspective(45.0,
                                      (double)cam_params.image_size.x / (double)cam_params.image_size.y,
                                      1.0, 100.0);
  glm::ivec4 viewport(0, 0, cam_params.image_size.x, cam_params.image_size.y);
  return glm::project(p, Mview, Mproj, viewport);
}

template <typename VecType>
double l1_norm(const VecType& u, const VecType& v) {
  double d = glm::distance(u, v);
  return sqrt(d);
};

template <typename VecType>
double l2_norm(const VecType& u, const VecType& v) {
  return glm::distance(u, v);
};

struct PoseCostFunction {
  PoseCostFunction(const MultilinearModel &model,
                   const Constraint2D &constraint,
                   const CameraParameters &cam_params)
    : model(model), constraint(constraint), cam_params(cam_params) {}

  bool operator()(const double* const params, double* residual) const {
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);

    auto Rmat = glm::eulerAngleYXZ(params[0], params[1], params[2]);
    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(params[3], params[4], params[5]));
    glm::dmat4 Mview = Tmat * Rmat;

    /// @todo Create projection matrix using camera focal length
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    residual[0] = l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    return true;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
};

struct PositionCostFunction {
  PositionCostFunction(const MultilinearModel &model,
                   const Constraint2D &constraint,
                   const CameraParameters &cam_params)
    : model(model), constraint(constraint), cam_params(cam_params) {}

  bool operator()(const double* const params, double* residual) const {
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);

    glm::dmat4 Mview = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(params[0], params[1], params[2]));

    /// @todo Create projection matrix using camera focal length
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    residual[0] = l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    return true;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
};

struct IdentityCostFunction {
  IdentityCostFunction(const MultilinearModel& model,
                       const Constraint2D& constraint,
                       int params_length,
                       const glm::mat4& Mview,
                       const CameraParameters& cam_params)
    : model(model), constraint(constraint),
      params_length(params_length),
      Mview(Mview), cam_params(cam_params) {}

  bool operator()(const double* const* wid, double* residual) const {
    // Apply the weight vector to the model
    model.UpdateTMWithTM1(Map<const VectorXd>(wid[0], params_length).eval());

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                 Mview,
                                 cam_params);
    // Compute residual
    residual[0] = l1_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview;
  CameraParameters cam_params;
};

struct ExpressionCostFunction {
  ExpressionCostFunction(const MultilinearModel& model,
                         const Constraint2D& constraint,
                         int params_length,
                         const glm::dmat4& Mview,
                         const CameraParameters& cam_params)
    : model(model), constraint(constraint), params_length(params_length),
      Mview(Mview), cam_params(cam_params) {}

  bool operator()(const double* const* wexp, double* residual) const {
    VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();

    // Apply the weight vector to the model
    model.UpdateTMWithTM0(wexp_vec);

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    //cout << p.x << ", " << p.y << ", " << p.z << endl;
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    // Compute residual
    residual[0] = l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview;
  CameraParameters cam_params;
};

struct ExpressionCostFunction_FACS {
  ExpressionCostFunction_FACS(const MultilinearModel& model,
                         const Constraint2D& constraint,
                         int params_length,
                         const glm::dmat4& Mview,
                         const MatrixXd& Uexp,
                         const CameraParameters& cam_params)
    : model(model), constraint(constraint), params_length(params_length),
      Mview(Mview), Uexp(Uexp), cam_params(cam_params) {}

  bool operator()(const double* const* wexp, double* residual) const {
    VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();
    VectorXd weights = (wexp_vec.transpose() * Uexp).eval();

    // Apply the weight vector to the model
    model.UpdateTMWithTM0(weights);

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    //cout << p.x << ", " << p.y << ", " << p.z << endl;
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);
    // Compute residual
    double dx = q.x - constraint.data.x;
    double dy = q.y - constraint.data.y;
    residual[0] = sqrt((dx * dx + dy * dy) * constraint.weight);
    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview;
  const MatrixXd& Uexp;
  CameraParameters cam_params;
};

struct PriorCostFunction {
  PriorCostFunction(const VectorXd& prior_vec, const MatrixXd& inv_cov_mat, double weight)
    : prior_vec(prior_vec), inv_cov_mat(inv_cov_mat), weight(weight) {}

  bool operator()(const double* const* w, double* residual) const {
    const int params_length = prior_vec.size();
    VectorXd diff = (Map<const VectorXd>(w[0], params_length) - prior_vec).eval();

    // Simply Mahalanobis distance between w and prior_vec
    residual[0] = sqrt(fabs(weight * diff.transpose() * (inv_cov_mat * diff)));
    return true;
  }

  const VectorXd& prior_vec;
  const MatrixXd& inv_cov_mat;
  double weight;
};

struct ExpressionPriorCostFunction {
  ExpressionPriorCostFunction(const VectorXd& prior_vec, const MatrixXd& inv_cov_mat,
                    const MatrixXd& Uexp, double weight)
    : prior_vec(prior_vec), inv_cov_mat(inv_cov_mat), Uexp(Uexp), weight(weight) {}

  bool operator()(const double* const* w, double* residual) const {
    const int params_length = 47;
    VectorXd wexp_vec = Map<const VectorXd>(w[0], 47).eval();
    VectorXd diff = (wexp_vec.transpose() * Uexp - prior_vec).eval();

    // Simply Mahalanobis distance between w and prior_vec
    residual[0] = sqrt(fabs(weight * diff.transpose() * (inv_cov_mat * diff)));
    return true;
  }

  const VectorXd& prior_vec;
  const MatrixXd& inv_cov_mat;
  const MatrixXd& Uexp;
  double weight;
};

#endif // COSTFUNCTIONS_H

