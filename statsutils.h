#ifndef STATS_UTILS_H
#define STATS_UTILS_H

#include "common.h"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

namespace StatsUtils {
static MatrixXd cov(const MatrixXd& mat) {
  MatrixXd centered = mat.rowwise() - mat.colwise().mean();
  MatrixXd cov_mat = (centered.adjoint() * centered) / double(mat.rows() - 1);
  return cov_mat;
};

static MatrixXd corr(const MatrixXd& mat) {
  MatrixXd cov_mat = cov(mat);
  MatrixXd corr_mat = MatrixXd::Zero(cov_mat.rows(), cov_mat.cols());
  for(int i=0;i<cov_mat.rows();++i) {
    for(int j=0;j<cov_mat.cols();++j) {
      corr_mat(i, j) = corr_mat(j, i) = cov_mat(i, j) / sqrt(cov_mat(i, i) * cov_mat(j, j));
    }
  }
  return corr_mat;
};

static MatrixXd dist(const MatrixXd& mat) {
  const int nsamples = mat.rows();
  MatrixXd dist_mat = MatrixXd::Zero(nsamples, nsamples);

  for(int i=0;i<nsamples;++i) {
    for(int j=i+1;j<nsamples;++j) {
      dist_mat(i, j) = dist_mat(j, i) = (mat.row(i) - mat.row(j)).norm();
    }
  }
  return dist_mat;
}

static MatrixXd normalize(const MatrixXd& mat) {
  const double max_val = mat.maxCoeff();
  const double min_val = mat.minCoeff();
  const double diff_val = max_val - min_val;

  const double DIFF_THRESHOLD = 1e-16;
  if(diff_val <= DIFF_THRESHOLD) {
    cerr << "Near-zero matrix. Not normalized." << endl;
    return mat;
  }

  MatrixXd normalized_mat(mat.rows(), mat.cols());
  for(int i=0;i<mat.rows();++i) {
    for(int j=0;j<mat.cols();++j) {
      normalized_mat(i, j) = (mat(i, j) - min_val) / diff_val;
    }
  }
  return normalized_mat;
}

static vector<int> FindConsistentSet(const MatrixXd& identity_weights,
                                     double radius) {
#if 1
  // Compute Pearson's correlation among identity weights
  MatrixXd metric_mat = StatsUtils::corr(identity_weights);
#else
  // Compute normalized Eucledian distance among identity weights
  MatrixXd metric_mat = MatrixXd::Ones(num_images, num_images) -
    StatsUtils::normalize(StatsUtils::dist(identity_weights.transpose()));
#endif

  // Pick a coherent subset

  // Compute the centroid of the coherent subset

  // Find the consistent set using the centroid and radius

  // @TODO Use the input set for the moment. Need to work on this later.
  vector<int> consistent_set;
  for(int i=0;i<identity_weights.cols();++i) consistent_set.push_back(i);

  return consistent_set;
}
}

#endif
