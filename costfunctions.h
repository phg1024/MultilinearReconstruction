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

#include "ceres/ceres.h"

inline glm::dvec3 ProjectPoint_ref(const glm::dvec3 &p, const glm::dmat4 &Mview,
                                   const CameraParameters &cam_params) {
  const double fovy = cam_params.fovy;
  const double aspect_ratio = static_cast<double>(cam_params.image_size.x) /
                              static_cast<double>(cam_params.image_size.y);
  const double top = 1.0;
  const double near = top / tan(fovy * 0.5), far = cam_params.far;

  glm::dmat4 Mproj = glm::perspective(fovy, aspect_ratio, near, far);

  // The projection matrix should be
  //   n/r, 0, 0, 0
  //   0, n/t, 0, 0
  //   0, 0, -(f+n)/(f-n), -2fn/(f-n)
  //   0, 0, -1, 0
  //
  // Therefore, if we assume f is infinite we have the following projection matrix
  //   n/r, 0, 0, 0
  //   0, n/t, 0, 0
  //   0, 0, -(f+n)/(f-n), -2fn/(f-n)
  //   0, 0, -1, 0
  //
  // Note: tan(fovy/2) = t/n

  glm::dmat4 Mproj_ref = glm::dmat4();

  glm::ivec4 viewport(0, 0, cam_params.image_size.x, cam_params.image_size.y);

  // glm::project
  // P = (p, 1.0)
  // P = Mview * P
  // P = Mproj * P
  // => P.x = P.x * n / r
  // => P.y = P.y * n / t
  // => P.z = -(f+n)/(f-n)*P.z - 2 * f * n / (f-n)
  // => P.w = -P.z
  // P = P / P.w
  // => P.x = -n / r * P.x / P.z
  // => P.y = -n / t * P.y / P.z
  // => P.z = -1.0 + 2 * n / P.z

  // P = P * 0.5 + 0.5
  // => P.x = -0.5 * n / r * P.x / P.z + 0.5
  // => P.y = -0.5 * n / t * P.y / P.z + 0.5
  // P.x = P.x * image_size_x + principle_x
  // P.y = P.y * image_size_y + principle_y
  // => P.x = -0.5 * n / r * image_size_x * P.x / P.z + 0.5 * image_size_x
  // => P.y = -0.5 * n / t * image_size_y * P.y / P.z + 0.5 * image_size_y

  return glm::project(p, Mview, Mproj, viewport);
}

inline glm::dvec3 ProjectPoint(const glm::dvec3 &p, const glm::dmat4 &Mview,
                               const CameraParameters &cam_params) {
  // use a giant canvas: r = image_size_x, t = image_size_y
  // then focal length = near plane z = 1.0

  // View transform
  glm::dvec4 P = Mview * glm::dvec4(p.x, p.y, p.z, 1.0);

  const double far = cam_params.far;
  const double near = cam_params.focal_length;
  const double top = near * tan(0.5 * cam_params.fovy);
  const double aspect_ratio = cam_params.image_size.x / cam_params.image_size.y;
  const double right = top * aspect_ratio;

  // Projection transform
  P.w = -P.z;
  P.x = near / right * P.x;
  P.y = near / top * P.y;
  P.z =
    -(far + near) / (far - near) * P.z - 2.0 * far * near / (far - near) * P.w;

  P /= P.w;

  P = 0.5 * P + 0.5;

  P.x = P.x * cam_params.image_size.x;
  P.y = P.y * cam_params.image_size.y;

  return glm::dvec3(P.x, P.y, P.z);
}

template<typename VecType>
double l1_norm(const VecType &u, const VecType &v) {
  double d = glm::distance(u, v);
  return sqrt(d);
};

template<typename VecType>
double l2_norm(const VecType &u, const VecType &v) {
  return glm::distance(u, v);
};

struct PoseCostFunction {
  PoseCostFunction(const MultilinearModel &model,
                   const Constraint2D &constraint,
                   const CameraParameters &cam_params)
    : model(model), constraint(constraint), cam_params(cam_params) { }

  bool operator()(const double *const params, double *residual) const {
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);

    auto Rmat = glm::eulerAngleYXZ(params[0], params[1], params[2]);
    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(params[3], params[4],
                                                params[5]));
    glm::dmat4 Mview = Tmat * Rmat;

    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    residual[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    return true;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
};

struct PoseRegularizationTerm {
  PoseRegularizationTerm(double weight) : weight(weight) {}

  bool operator()(const double *const params, double *residual) const {
    residual[0] = params[1] * weight;
    return true;
  }

  double weight;
};

// differential of euler angle functions
namespace glm {
  template<typename T>
  GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> dEulerAngleX
    (
      T const &angleX
    ) {
    T cosX = glm::cos(angleX);
    T sinX = glm::sin(angleX);

    return tmat4x4<T, defaultp>(
      T(0), T(0), T(0), T(0),
      T(0), -sinX, cosX, T(0),
      T(0), -cosX, -sinX, T(0),
      T(0), T(0), T(0), T(0));
  }

  template<typename T>
  GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> dEulerAngleY
    (
      T const &angleY
    ) {
    T cosY = glm::cos(angleY);
    T sinY = glm::sin(angleY);

    return tmat4x4<T, defaultp>(
      -sinY, T(0), -cosY, T(0),
      T(0), T(0), T(0), T(0),
      cosY, T(0), -sinY, T(0),
      T(0), T(0), T(0), T(0));
  }

  template<typename T>
  GLM_FUNC_QUALIFIER tmat4x4<T, defaultp> dEulerAngleZ
    (
      T const &angleZ
    ) {
    T cosZ = glm::cos(angleZ);
    T sinZ = glm::sin(angleZ);

    return tmat4x4<T, defaultp>(
      -sinZ, cosZ, T(0), T(0),
      -cosZ, -sinZ, T(0), T(0),
      T(0), T(0), T(0), T(0),
      T(0), T(0), T(0), T(0));
  }
}

struct PoseCostFunction_analytic : public ceres::SizedCostFunction<2, 3, 3> {
  PoseCostFunction_analytic(const MultilinearModel &model,
                            const Constraint2D &constraint,
                            const CameraParameters &cam_params)
    : model(model), constraint(constraint), cam_params(cam_params) { }

  virtual bool Evaluate(double const *const *params,
                        double *residuals,
                        double **jacobians) const {
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);

    auto Ry = glm::eulerAngleY(params[0][0]);
    auto Rx = glm::eulerAngleX(params[0][1]);
    auto Rz = glm::eulerAngleZ(params[0][2]);

    auto Rmat = Ry * Rx * Rz;
    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(params[1][0], params[1][1],
                                                params[1][2]));
    glm::dmat4 Mview = Tmat * Rmat;

    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    residuals[0] = (q.x - constraint.data.x) * constraint.weight;
    residuals[1] = (q.y - constraint.data.y) * constraint.weight;

    // Now compute Jacobians
    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        glm::dvec4 P = Mview * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
        const double x0 = P.x, y0 = P.y, z0 = P.z;

        double dx = q.x - constraint.data.x;
        double dy = q.y - constraint.data.y;
        const double sx = cam_params.image_size.x;
        const double sy = cam_params.image_size.y;
        const double f = cam_params.focal_length;

//  Jocobian of projection-viewport transformation
//      double Jh[6] = {
//        -0.5 * sy * f / z0, 0, 0.5 * sy * f * x0 / (z0 * z0),
//        0, -0.5 * sy * f / z0, 0.5 * sy * f * y0 / (z0 * z0)
//      };

        const double inv_z0 = 1.0 / z0;
        const double common_factor = 0.5 * sy * f * inv_z0;

        auto dRx = glm::dEulerAngleX(params[0][1]);
        auto dRy = glm::dEulerAngleY(params[0][0]);
        auto dRz = glm::dEulerAngleZ(params[0][2]);

        auto dRyRxRz = dRy * Rx * Rz;
        auto RydRxRz = Ry * dRx * Rz;
        auto RyRxdRz = Ry * Rx * dRz;

        auto Py = dRyRxRz * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
        jacobians[0][0] =
          -Py.x * common_factor + Py.z * common_factor * x0 * inv_z0;
        jacobians[0][3] =
          -Py.y * common_factor + Py.z * common_factor * y0 * inv_z0;

        auto Px = RydRxRz * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
        jacobians[0][1] =
          -Px.x * common_factor + Px.z * common_factor * x0 * inv_z0;
        jacobians[0][4] =
          -Px.y * common_factor + Px.z * common_factor * y0 * inv_z0;

        auto Pz = RyRxdRz * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
        jacobians[0][2] =
          -Pz.x * common_factor + Pz.z * common_factor * x0 * inv_z0;
        jacobians[0][5] =
          -Pz.y * common_factor + Pz.z * common_factor * y0 * inv_z0;
      }

      if (jacobians[1] != NULL) {
        glm::dvec4 P = Mview * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
        const double x0 = P.x, y0 = P.y, z0 = P.z;

        double dx = q.x - constraint.data.x;
        double dy = q.y - constraint.data.y;
        const double sx = cam_params.image_size.x;
        const double sy = cam_params.image_size.y;
        const double f = cam_params.focal_length;

        const double inv_z0 = 1.0 / z0;
        const double common_factor = 0.5 * sy * f * inv_z0;
        jacobians[1][0] = -common_factor;
        jacobians[1][1] = 0;
        jacobians[1][2] = common_factor * x0 * inv_z0;
        jacobians[1][3] = 0;
        jacobians[1][4] = -common_factor;
        jacobians[1][5] = common_factor * y0 * inv_z0;
      }
    }
    return true;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
};

struct PositionCostFunction {
  PositionCostFunction(const MultilinearModel &model,
                       const Constraint2D &constraint,
                       const CameraParameters &cam_params,
                       double theta_z)
    : model(model), constraint(constraint), cam_params(cam_params),
      theta_z(theta_z) {}

  bool operator()(const double *const params, double *residual) const {
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);

    glm::dmat4 Rmat = glm::eulerAngleZ(theta_z);
    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(params[0], params[1],
                                                params[2]));
    glm::dmat4 Mview = Tmat * Rmat;

    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    residual[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    return true;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
  double theta_z;
};

struct PositionCostFunction_analytic : public ceres::SizedCostFunction<2, 3> {
  PositionCostFunction_analytic(const MultilinearModel &model,
                                const Constraint2D &constraint,
                                const CameraParameters &cam_params,
                                double theta_z)
    : model(model), constraint(constraint), cam_params(cam_params),
      theta_z(theta_z) {}

  virtual bool Evaluate(double const *const *params,
                        double *residuals,
                        double **jacobians) const {
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);

    glm::dmat4 Rmat = glm::eulerAngleZ(theta_z);
    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(params[0][0], params[0][1],
                                                params[0][2]));
    glm::dmat4 Mview = Tmat * Rmat;

    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    residuals[0] = (q.x - constraint.data.x) * constraint.weight;
    residuals[1] = (q.y - constraint.data.y) * constraint.weight;

    // Now compute Jacobians
    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);

      glm::dvec4 P = Mview * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
      const double x0 = P.x, y0 = P.y, z0 = P.z;

      double dx = q.x - constraint.data.x;
      double dy = q.y - constraint.data.y;
      const double sx = cam_params.image_size.x;
      const double sy = cam_params.image_size.y;
      const double f = cam_params.focal_length;

//      double Jh[6] = {
//        -0.5 * sy * f / z0, 0, 0.5 * sy * f * x0 / (z0 * z0),
//        0, -0.5 * sy * f / z0, 0.5 * sy * f * y0 / (z0 * z0)
//      };

      const double inv_z0 = 1.0 / z0;
      const double common_factor = 0.5 * sy * f * inv_z0;
      jacobians[0][0] = -common_factor;
      jacobians[0][1] = 0;
      jacobians[0][2] = common_factor * x0 * inv_z0;
      jacobians[0][3] = 0;
      jacobians[0][4] = -common_factor;
      jacobians[0][5] = common_factor * y0 * inv_z0;
    }
    return true;
  }

  MultilinearModel model;
  Constraint2D constraint;
  CameraParameters cam_params;
  double theta_z;
};

struct IdentityCostFunction {
  IdentityCostFunction(const MultilinearModel &model,
                       const Constraint2D &constraint,
                       int params_length,
                       const glm::mat4 &Mview,
                       const CameraParameters &cam_params)
    : model(model), constraint(constraint),
      params_length(params_length),
      Mview(Mview), cam_params(cam_params) { }

  bool operator()(const double *const *wid, double *residual) const {
    // Apply the weight vector to the model
    model.UpdateTMWithTM1(Map<const VectorXd>(wid[0], params_length).eval());

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                Mview,
                                cam_params);
    // Compute residual
    residual[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview;
  CameraParameters cam_params;
};

struct IdentityCostFunction_analytic : public ceres::CostFunction {
  IdentityCostFunction_analytic(const MultilinearModel &model,
                                const Constraint2D &constraint,
                                int params_length,
                                const glm::dmat4 &Mview,
                                const glm::dmat4 Rmat,
                                const CameraParameters &cam_params,
                                double weight = 1.0)
    : model(model), constraint(constraint),
      params_length(params_length),
      Mview(Mview), Rmat(Rmat), cam_params(cam_params), weight(weight) {
    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(params_length);
    set_num_residuals(1);
  }

  VectorXd jacobian_ref(double const *const *wid) const {
    VectorXd J_ref(params_length);
    const double epsilon = 1e-12;
    for (int i = 0; i < params_length; ++i) {
      VectorXd wid_vec = Map<const VectorXd>(wid[0], params_length);
      VectorXd wid_vec_m = wid_vec, wid_vec_p = wid_vec;
      wid_vec_m[i] -= epsilon * 0.5;
      wid_vec_p[i] += epsilon * 0.5;

      double residual_p, residual_m;

      {
        model.UpdateTMWithTM1(wid_vec_p);
        auto tm = model.GetTM();
        glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                    Mview,
                                    cam_params);
        residual_p =
          l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
      }

      {
        model.UpdateTMWithTM1(wid_vec_m);
        auto tm = model.GetTM();
        glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                    Mview,
                                    cam_params);
        residual_m =
          l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
      }

      J_ref[i] = (residual_p - residual_m) / epsilon;
    }
    return J_ref;
  }

  virtual bool Evaluate(double const *const *wid,
                        double *residuals,
                        double **jacobians) const {
    // Apply the weight vector to the model
    model.UpdateTMWithTM1(Map<const VectorXd>(wid[0], params_length).eval());

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                Mview,
                                cam_params);
    // Compute residual
    // residuals[0] = dot(p - q, p - q)^0.5;
    residuals[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight * weight;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      // J = Jh * R * tm1^T

      Vector2d fvec;
      fvec[0] = q.x - constraint.data.x;
      fvec[1] = q.y - constraint.data.y;

      const double scale_factor = 1.0 / std::pow(fvec.dot(fvec), 0.5);

      glm::dvec4 P = Mview * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
      const double x0 = P.x, y0 = P.y, z0 = P.z;

      const double sx = cam_params.image_size.x;
      const double sy = cam_params.image_size.y;
      const double f = cam_params.focal_length;

      const double inv_z0 = 1.0 / z0;
      const double common_factor = 0.5 * sy * f * inv_z0;
      MatrixXd Jh(2, 3);
      Jh(0, 0) = -common_factor;
      Jh(0, 1) = 0;
      Jh(0, 2) = common_factor * x0 * inv_z0;
      Jh(1, 0) = 0;
      Jh(1, 1) = -common_factor;
      Jh(1, 2) = common_factor * y0 * inv_z0;

      Matrix3d R;
      R(0, 0) = Rmat[0][0];
      R(0, 1) = Rmat[1][0];
      R(0, 2) = Rmat[2][0];
      R(1, 0) = Rmat[0][1];
      R(1, 1) = Rmat[1][1];
      R(1, 2) = Rmat[2][1];
      R(2, 0) = Rmat[0][2];
      R(2, 1) = Rmat[1][2];
      R(2, 2) = Rmat[2][2];

      // tm1 is a ndims_id x 3 matrix, where each row is x, y, z
      auto tm1 = model.GetTM1().GetData();
      auto J = (scale_factor * fvec.transpose() * Jh * R *
                tm1.transpose()).eval();

      for (int i = 0; i < params_length; ++i) jacobians[0][i] = J(0, i) * weight;
    }

    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview, Rmat;
  CameraParameters cam_params;
  double weight;
};

struct ExpressionCostFunction {
  ExpressionCostFunction(const MultilinearModel &model,
                         const Constraint2D &constraint,
                         int params_length,
                         const glm::dmat4 &Mview,
                         const CameraParameters &cam_params)
    : model(model), constraint(constraint), params_length(params_length),
      Mview(Mview), cam_params(cam_params) { }

  bool operator()(const double *const *wexp, double *residual) const {
    VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();

    // Apply the weight vector to the model
    model.UpdateTMWithTM0(wexp_vec);

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    //cout << p.x << ", " << p.y << ", " << p.z << endl;
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    // Compute residual
    residual[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview;
  CameraParameters cam_params;
};

struct ExpressionCostFunction_analytic : public ceres::CostFunction {
  ExpressionCostFunction_analytic(const MultilinearModel &model,
                                  const Constraint2D &constraint,
                                  int params_length,
                                  const glm::dmat4 &Mview,
                                  const glm::dmat4 &Rmat,
                                  const CameraParameters &cam_params)
    : model(model), constraint(constraint), params_length(params_length),
      Mview(Mview), Rmat(Rmat), cam_params(cam_params) {
    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(params_length);
    set_num_residuals(1);
  }

  bool Evaluate(const double *const *wexp, double *residuals,
                double **jacobians) const {
    VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();

    // Apply the weight vector to the model
    model.UpdateTMWithTM0(wexp_vec);

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    //cout << p.x << ", " << p.y << ", " << p.z << endl;
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);

    // Compute residual
    residuals[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      // J = Jh * R * tm1^T

      Vector2d fvec;
      fvec[0] = q.x - constraint.data.x;
      fvec[1] = q.y - constraint.data.y;

      const double scale_factor = 1.0 / std::sqrt(fvec.dot(fvec));

      glm::dvec4 P = Mview * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
      const double x0 = P.x, y0 = P.y, z0 = P.z;

      const double sx = cam_params.image_size.x;
      const double sy = cam_params.image_size.y;
      const double f = cam_params.focal_length;

      const double inv_z0 = 1.0 / z0;
      const double common_factor = 0.5 * sy * f * inv_z0;
      MatrixXd Jh(2, 3);
      Jh(0, 0) = -common_factor;
      Jh(0, 1) = 0;
      Jh(0, 2) = common_factor * x0 * inv_z0;
      Jh(1, 0) = 0;
      Jh(1, 1) = -common_factor;
      Jh(1, 2) = common_factor * y0 * inv_z0;

      Matrix3d R;
      R(0, 0) = Rmat[0][0];
      R(0, 1) = Rmat[1][0];
      R(0, 2) = Rmat[2][0];
      R(1, 0) = Rmat[0][1];
      R(1, 1) = Rmat[1][1];
      R(1, 2) = Rmat[2][1];
      R(2, 0) = Rmat[0][2];
      R(2, 1) = Rmat[1][2];
      R(2, 2) = Rmat[2][2];

      // tm1 is a ndims_id x 3 matrix, where each row is x, y, z
      auto tm0 = model.GetTM0().GetData();
      auto J = (scale_factor * fvec.transpose() * Jh * R *
                tm0.transpose()).eval();

      for (int i = 0; i < params_length; ++i) jacobians[0][i] = J(0, i);
    }

    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview, Rmat;
  CameraParameters cam_params;
};

struct ExpressionCostFunction_FACS {
  ExpressionCostFunction_FACS(const MultilinearModel &model,
                              const Constraint2D &constraint,
                              int params_length,
                              const glm::dmat4 &Mview,
                              const MatrixXd &Uexp,
                              const CameraParameters &cam_params)
    : model(model), constraint(constraint), params_length(params_length),
      Mview(Mview), Uexp(Uexp), cam_params(cam_params) { }

  bool operator()(const double *const *wexp, double *residual) const {
#if 0
    VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();
#else
    VectorXd wexp_vec(params_length);
    wexp_vec.bottomRows(params_length-1) = Map<const VectorXd>(wexp[0], params_length - 1).eval();
    wexp_vec[0] = 1.0 - wexp_vec.bottomRows(params_length-1).sum();
#endif
    VectorXd weights = (wexp_vec.transpose() * Uexp).eval();

    // Apply the weight vector to the model
    model.UpdateTMWithTM0(weights);

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);
    // Compute residual
    residual[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview;
  const MatrixXd &Uexp;
  CameraParameters cam_params;
};

struct ExpressionCostFunction_FACS_analytic : public ceres::CostFunction {
  ExpressionCostFunction_FACS_analytic(const MultilinearModel &model,
                                       const Constraint2D &constraint,
                                       int params_length,
                                       const glm::dmat4 &Mview,
                                       const glm::dmat4 &Rmat,
                                       const MatrixXd &Uexp,
                                       const CameraParameters &cam_params)
    : model(model), constraint(constraint), params_length(params_length),
      Mview(Mview), Rmat(Rmat), Uexp(Uexp), cam_params(cam_params) {
    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(params_length - 1);
    set_num_residuals(1);
  }

  // TODO update the reference function
  VectorXd jacobian_ref(double const *const *wexp) const {
    VectorXd J_ref(params_length-1);
    const double epsilon = 1e-12;
    for (int i = 1; i < params_length; ++i) {
      #if 0
          VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();
      #else
          VectorXd wexp_vec(params_length);
          wexp_vec.bottomRows(params_length-1) = Map<const VectorXd>(wexp[0], params_length - 1).eval();
          wexp_vec[0] = 1.0 - wexp_vec.bottomRows(params_length-1).sum();
      #endif

      VectorXd wexp_vec_m = wexp_vec, wexp_vec_p = wexp_vec;
      wexp_vec_m[i] -= epsilon * 0.5;
      wexp_vec_p[i] += epsilon * 0.5;

      double residual_p, residual_m;

      {
        model.UpdateTMWithTM0((wexp_vec_p.transpose() * Uexp).eval());
        auto tm = model.GetTM();
        glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                    Mview,
                                    cam_params);
        residual_p =
          l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
      }

      {
        model.UpdateTMWithTM0((wexp_vec_m.transpose() * Uexp).eval());
        auto tm = model.GetTM();
        glm::dvec3 q = ProjectPoint(glm::dvec3(tm[0], tm[1], tm[2]),
                                    Mview,
                                    cam_params);
        residual_m =
          l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;
      }

      J_ref[i-1] = (residual_p - residual_m) / epsilon;
    }
    return J_ref;
  }

  bool Evaluate(const double *const *wexp, double *residuals,
                double **jacobians) const {
#if 0
    VectorXd wexp_vec = Map<const VectorXd>(wexp[0], params_length).eval();
#else
    VectorXd wexp_vec(params_length);
    wexp_vec.bottomRows(params_length-1) = Map<const VectorXd>(wexp[0], params_length - 1).eval();
    wexp_vec[0] = 1.0 - wexp_vec.bottomRows(params_length-1).sum();
#endif

    VectorXd weights = (wexp_vec.transpose() * Uexp).eval();

    // Apply the weight vector to the model
    model.UpdateTMWithTM0(weights);

    // Project the point to image plane
    auto tm = model.GetTM();
    glm::dvec3 p(tm[0], tm[1], tm[2]);
    glm::dvec3 q = ProjectPoint(p, Mview, cam_params);
    // Compute residual
    residuals[0] =
      l2_norm(glm::dvec2(q.x, q.y), constraint.data) * constraint.weight;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      // J = Jh * R * tm1^T

      Vector2d fvec;
      fvec[0] = q.x - constraint.data.x;
      fvec[1] = q.y - constraint.data.y;

      const double scale_factor = 1.0 / std::sqrt(fvec.dot(fvec));

      glm::dvec4 P = Mview * glm::dvec4(tm[0], tm[1], tm[2], 1.0);
      const double x0 = P.x, y0 = P.y, z0 = P.z;

      const double sx = cam_params.image_size.x;
      const double sy = cam_params.image_size.y;
      const double f = cam_params.focal_length;

      const double inv_z0 = 1.0 / z0;
      const double common_factor = 0.5 * sy * f * inv_z0;
      MatrixXd Jh(2, 3);
      Jh(0, 0) = -common_factor;
      Jh(0, 1) = 0;
      Jh(0, 2) = common_factor * x0 * inv_z0;
      Jh(1, 0) = 0;
      Jh(1, 1) = -common_factor;
      Jh(1, 2) = common_factor * y0 * inv_z0;

      Matrix3d R;
      R(0, 0) = Rmat[0][0];
      R(0, 1) = Rmat[1][0];
      R(0, 2) = Rmat[2][0];
      R(1, 0) = Rmat[0][1];
      R(1, 1) = Rmat[1][1];
      R(1, 2) = Rmat[2][1];
      R(2, 0) = Rmat[0][2];
      R(2, 1) = Rmat[1][2];
      R(2, 2) = Rmat[2][2];

      MatrixXd D = MatrixXd::Zero(params_length, params_length-1);
      for(int i=0;i<params_length-1;++i) {
        D(0, i) = -1.0;
        D(i+1, i) = 1.0;
      }

      // tm0 is a ndims_exp x 3 matrix, where each row is x, y, z
      auto tm0 = model.GetTM0().GetData();
      auto J = (scale_factor * fvec.transpose() * Jh * R * tm0.transpose() *
                Uexp.transpose() * D).eval();

      for (int i = 0; i < params_length - 1; ++i) jacobians[0][i] = J(0, i);
    }

    return true;
  }

  mutable MultilinearModel model;
  int params_length;

  Constraint2D constraint;
  glm::dmat4 Mview, Rmat;
  const MatrixXd &Uexp;
  CameraParameters cam_params;
};

struct PriorCostFunction {
  PriorCostFunction(const VectorXd &prior_vec, const MatrixXd &inv_cov_mat,
                    double weight)
    : prior_vec(prior_vec), inv_cov_mat(inv_cov_mat), weight(weight) { }

  bool operator()(const double *const *w, double *residual) const {
    const int params_length = prior_vec.size();
    VectorXd diff = (Map<const VectorXd>(w[0], params_length) -
                     prior_vec).eval();

    // Simply Mahalanobis distance between w and prior_vec
    residual[0] = sqrt(fabs(weight * diff.transpose() * (inv_cov_mat * diff)));
    return true;
  }

  const VectorXd &prior_vec;
  const MatrixXd &inv_cov_mat;
  double weight;
};

struct ExpressionRegularizationCostFunction {
  ExpressionRegularizationCostFunction(const VectorXd &prior_vec,
                                       const MatrixXd &inv_cov_mat,
                                       const MatrixXd &Uexp, double weight)
    : prior_vec(prior_vec), inv_cov_mat(inv_cov_mat), Uexp(Uexp),
      weight(weight) { }

  bool operator()(const double *const *w, double *residual) const {
    const int params_length = 47;
#if 0
    VectorXd wexp_vec = Map<const VectorXd>(w[0], params_length).eval();
#else
    VectorXd wexp_vec(params_length);
    wexp_vec.bottomRows(params_length-1) = Map<const VectorXd>(w[0], params_length - 1).eval();
    wexp_vec[0] = 1.0 - wexp_vec.bottomRows(params_length-1).sum();
#endif

    VectorXd diff = (Uexp.transpose() * wexp_vec - prior_vec).eval();

    // Simply Mahalanobis distance between w and prior_vec
    residual[0] = sqrt(fabs(weight * diff.transpose() * (inv_cov_mat * diff)));

    return true;
  }

  const VectorXd &prior_vec;
  const MatrixXd &inv_cov_mat;
  const MatrixXd &Uexp;
  double weight;
};

struct ExpressionRegularizationTerm {
  ExpressionRegularizationTerm(double weight) : weight(weight) {}

  bool operator()(const double *const *w, double *residual) const {
    const int params_length = 47;
    for(int i=0;i<params_length-1;++i) {
      residual[i] = sqrt(fabs(w[0][i])) * weight;
    }
    return true;
  }

  double weight;
};

struct IdentityRegularizationTerm {
  IdentityRegularizationTerm(double weight) : weight(weight) {}

  bool operator()(const double *const *w, double *residual) const {
    const int params_length = 50;
    residual[0] = 0;
    for(int i=0;i<params_length-1;++i) {
      residual[0] += w[0][i]*w[0][i];
    }
    residual[0] *= weight;
    return true;
  }

  double weight;
};

#endif // COSTFUNCTIONS_H
