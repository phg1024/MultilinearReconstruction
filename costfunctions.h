#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H

#include "common.h"
#include "constraints.h"
#include "multilinearmodel.h"

#include "glm/glm.hpp"
#include "glm/gtx/euler_angles.hpp"
#include <eigen3/Eigen/Dense>
using namespace Eigen;

struct CameraParameters {
  glm::dvec2 focal_length;
  glm::dvec2 image_plane_center;
  glm::dvec2 image_size;
};

glm::dvec2 ProjectPoint(const glm::dvec4& p, const CameraParameters& cam_params) {
  double X = p[0], Y = p[1], Z = p[2];
  double invZ = 1.0 / Z;
  double x = cam_params.focal_length.x * X * invZ;
  double y = cam_params.focal_length.y * Y * invZ;
  x = x * cam_params.image_size.x + cam_params.image_plane_center.x;
  y = y * cam_params.image_size.y + cam_params.image_plane_center.y;
  return glm::vec2(x, y);
}

struct PoseCostFunction {
  PoseCostFunction() {

  }

  bool operator()(const double* const params, double* residual) {
    double yaw = params[0], pitch = params[1], roll = params[2];
    glm::dvec4 T(params[3], params[4], params[5], 1.0);
    auto R = glm::eulerAngleYXZ(yaw, pitch, roll);

    auto tm = model.GetTM();
    glm::dvec4 P0(tm[0], tm[1], tm[2], 1.0);
    glm::dvec4 P = R * P0 + T;
    glm::dvec2 uv = ProjectPoint(P + T, cam_params);

    residual[0] = uv.x - constraint.data.x;
    residual[1] = uv.y - constraint.data.y;

    return 0;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
};

struct IdentityCostFunction {
  IdentityCostFunction(const Constraint2D& constraint,
                       const MultilinearModel& fullmodel,
                       int params_length,
                       const CameraParameters& cam_params)
    : constraint(constraint), params_length(params_length),
      cam_params(cam_params) {
    // Create a projection of the model to this point only
    model = fullmodel.project(vector<int>(1, constraint.vidx));
  }

  bool operator()(const double* const wid, double* residual) {
    // Apply the weight vector to the model
    model.UpdateTMWithTM1(Map<const VectorXd>(wid, params_length));

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec2 uv = ProjectPoint(glm::dvec4(tm[0], tm[1], tm[2], 1.0),
                              cam_params);
    // Compute residual
    residual[0] = uv.x - constraint.data.x;
    residual[0] = uv.y - constraint.data.y;
  }

  MultilinearModel model;
  int params_length;

  Constraint2D constraint;

  CameraParameters cam_params;
};

#endif // COSTFUNCTIONS_H

