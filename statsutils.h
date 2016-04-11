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

static VectorXd mean(const MatrixXd& mat, int dim) {
  switch(dim) {
    case 1: {
      // column wise mean
      VectorXd m = VectorXd::Zero(mat.cols());
      for(int i=0;i<mat.cols();++i) {
        m += mat.col(i);
      }
      m = m / static_cast<double>(mat.cols());
      return m;
    }
    case 2: {
      // row wise mean
      // column wise mean
      VectorXd m = VectorXd::Zero(mat.rows());
      for(int i=0;i<mat.cols();++i) {
        m += mat.row(i).transpose();
      }
      m = m / static_cast<double>(mat.rows());
      return m;

    }
  }
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

static vector<int> FindConsistentSet_kMeans(const MatrixXd& x, int k) {
  // Run a k-means to separate the good set and others

  // Compute the centroid of the good set

  // Pick k closest as the consistent set

  return vector<int>();
}

static vector<int> FindConsistentSet(const MatrixXd& x, double h, int k,
                                     VectorXd* centroid_out=nullptr) {
  // Meanshift until converged
  int ndims = x.rows(), nsamples = x.cols();
  MatrixXd m(ndims, nsamples);
  MatrixXd y = x;
  const double th = 1e-6;
  const int max_iters = 100;
  bool done = false;

  int iters = 0;
  double ms = 0;
  while(!done && iters < max_iters) {
    for(int i=0;i<nsamples;++i) {
      double gsum = 0;
      VectorXd yi = VectorXd::Zero(ndims);
      for(int j=0;j<nsamples;++j) {
        if(j==i) continue;
        else {
          VectorXd dj = (y.col(i) - x.col(j))/h;
          double gj = exp(-dj.transpose() * dj);
          gsum += gj;
          yi += gj * x.col(j);
        }
      }
      m.col(i) = yi / (gsum + 1e-8);
    }

    ms = m.maxCoeff();

    if(ms < th) {
      cout << "ms = " << ms << endl;
      done = true;
    } else {
      y = m;
      ++iters;
      cout << "iteration " << iters << ": " << ms << endl;
    }
  }

  // Find the highest density cluster, compute its centroid
  vector<double> d(nsamples, 0);
  for(int i=0;i<nsamples;++i) {
    for(int j=0;j<nsamples;++j) {
      d[i] += exp(-(y.col(i) - y.col(j)).squaredNorm());
    }
  }

  double max_d = 0;
  int max_idx = -1;
  for(int i=0;i<nsamples;++i) {
    if(d[i] > max_d) {
      max_d = d[i];
      max_idx = i;
    }
  }

  VectorXd centroid = y.col(max_idx);
  if(centroid_out != nullptr) {
    // Write the centroid to the output
    *centroid_out = centroid;
  }

  // Compute the distance between the centroid and each input point
  vector<pair<int, double>> dists(nsamples);
  for(int i=0;i<nsamples;++i) {
    dists[i] = make_pair(i, (y.col(i) - centroid).norm());
  }

  std::sort(dists.begin(), dists.end(),
            [](const pair<int, double>& a, const pair<int, double>& b) {
              return a.second < b.second;
            });

  // Pick the k nearest points as the consistent set
  vector<int> consistent_set;
  for(int i=0;i<k;++i) {
    consistent_set.push_back(dists[i].first);
  }

  return consistent_set;
}

cv::Mat MeanShiftSegmentation(const cv::Mat& x, double hs, double hr, double th) {
  cout << "Mean shift segmentation." << endl;
  int height = x.rows, width = x.cols;
  bool done = false;
  int iters = 0;

  // weight map for color space
  vector<double> weight_map(255*255+1, 0);
  for(int i=0;i<255*255+1;++i) {
    weight_map[i] = exp(-i/(hr*hr));
  }

  cv::Mat y = x.clone();

  while(!done) {
    ++iters;

    cv::Mat weightAccum(height, width, CV_64F, 0.0);
    cv::Mat yAccum(height, width, CV_64FC3, 0.0);
    cv::Mat xThis(height, width, CV_64FC3);

    for(int i=-hs;i<=hs;++i) {
      for(int j=-hs;j<=hs;++j) {
        if(i==0 && j==0) continue;
        double spatialKernel = 1.0;

        #pragma omp parallel for
        for(int r=0;r<height;++r) {
          int r0 = r + i;
          if(r0<0) r0 += height;
          if(r0>=height) r0 -= height;

          for(int c=0;c<width;++c) {
            int c0 = c + j;
            if(c0<0) c0 += width;
            if(c0>=width) c0 -= width;

            xThis.at<cv::Vec3d>(r, c) = x.at<cv::Vec3d>(r0, c0);
          }
        }

        cv::Mat xDiffSq = y - xThis;

        #pragma omp parallel for
        for(int r=0;r<height;++r) {
          for(int c=0;c<width;++c) {
            cv::Vec3d pix = xDiffSq.at<cv::Vec3d>(r, c);
            xDiffSq.at<cv::Vec3d>(r, c) = cv::Vec3d(pix[0]*pix[0] + 1, pix[1]*pix[1] + 1, pix[2]*pix[2] + 1);
          }
        }

        cv::Mat intensityKernel(height, width, CV_64F);
        #pragma omp parallel for
        for(int r=0;r<height;++r) {
          for(int c=0;c<width;++c) {
            cv::Vec3d pix = xDiffSq.at<cv::Vec3d>(r, c);
            intensityKernel.at<double>(r, c) = weight_map[pix[0]] * weight_map[pix[1]] * weight_map[pix[2]];
          }
        }

        cv::Mat weightThis = intensityKernel * spatialKernel;

        weightAccum += weightThis;
        #pragma omp parallel for
        for(int r=0;r<height;++r) {
          for(int c=0;c<width;++c) {
            cv::Vec3d pix = xThis.at<cv::Vec3d>(r, c);
            yAccum.at<cv::Vec3d>(r, c) += pix * weightThis.at<double>(r, c);
          }
        }
      }
    }

    cv::Mat yThis = yAccum.clone();
    #pragma omp parallel for
    for(int r=0;r<height;++r) {
      for(int c=0;c<width;++c) {
        yThis.at<cv::Vec3d>(r, c) /= (weightAccum.at<double>(r, c) + 1e-16);
      }
    }

    double yMS = 0;
    //#pragma omp parallel for
    for(int r=0;r<height;++r) {
      for(int c=0;c<width;++c) {
        cv::Vec3d p1 = yThis.at<cv::Vec3d>(r, c);
        cv::Vec3d p0 = y.at<cv::Vec3d>(r, c);
        yMS += fabs(round(p1[0]) - round(p0[0]));
        yMS += fabs(round(p1[1]) - round(p0[1]));
        yMS += fabs(round(p1[2]) - round(p0[2]));
      }
    }
    yMS /= (height*width*3);

    cout << "iteration " << iters << ": " << yMS << endl;

    y = yThis.clone();
    #pragma omp parallel for
    for(int r=0;r<height;++r) {
      for(int c=0;c<width;++c) {
        cv::Vec3d pix = yThis.at<cv::Vec3d>(r, c);
        y.at<cv::Vec3d>(r, c) = cv::Vec3d(round(pix[0]), round(pix[1]), round(pix[2]));
      }
    }

    if(yMS<=th) {
      done = true;
    }
  }

  return y;
}
}

#endif
