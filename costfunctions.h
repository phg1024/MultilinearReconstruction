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

glm::dvec2 ProjectPoint(const glm::dvec4& p, const CameraParameters& cam_params) {
#if 0
  double X = p[0], Y = p[1], Z = p[2];
  glm::dmat4 Mproj(cam_params.focal_length.x, 0., 0., 0.,
                   0., cam_params.focal_length.y, 0., 0.,
                   0., 0., 1., 0.,
                   0., 0., 0., 1.);

  glm::dvec4 UV = Mproj * p;
  UV = UV / UV.z;
  double x = UV.x;
  double y = UV.y;

  x = x * cam_params.image_size.x + cam_params.image_plane_center.x;
  y = y * cam_params.image_size.y + cam_params.image_plane_center.y;
  return glm::vec2(x, y);
#else

#endif
}

struct PoseCostFunction {
  PoseCostFunction(const MultilinearModel &model,
                   const Constraint2D &constraint,
                   const CameraParameters &cam_params)
    : model(model), constraint(constraint), cam_params(cam_params) {}

  bool operator()(const double* const params, double* residual) const {
    double yaw = params[0], pitch = params[1], roll = params[2];
    glm::dvec4 T(params[3], params[4], params[5], 1.0);
    auto Rmat = glm::eulerAngleYXZ(yaw, pitch, roll);

    auto tm = model.GetTM();
    glm::dvec4 P0(tm[0], tm[1], tm[2], 1.0);

#if 0
    glm::dvec4 P = R * P0 + T;
    glm::dvec2 uv = ProjectPoint(P + T, cam_params);
#else
    double u, v, d;
    glm::dmat4 Mproj = glm::perspective(45.0,
                                        (double)cam_params.image_size.x / (double)cam_params.image_size.y,
                                        1.0, 10.0);
    int viewport[] = {0, 0, cam_params.image_size.x, cam_params.image_size.y};
    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0), glm::dvec3(T.x, T.y, T.z));
    glm::dmat4 MV = Tmat * Rmat;
    gluProject(tm[0], tm[1], tm[2], &MV[0][0], &Mproj[0][0], viewport, &u, &v, &d);
    glm::dvec2 uv(u, v);
#endif

    residual[0] = uv.x - constraint.data.x;
    residual[1] = uv.y - constraint.data.y;
    cout << "(" << uv.x << ", " << uv.y << ")"
         << " vs "
         << "(" << constraint.data.x << ", " << constraint.data.y << ")" << endl;

    return true;
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

