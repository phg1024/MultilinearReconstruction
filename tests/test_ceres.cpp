// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include <cstdlib>
#include <vector>
using namespace std;

// Data generated using the following octave code.
//   randn('seed', 23497);
//   m = 0.3;
//   c = 0.1;
//   x=[0:0.075:5];
//   y = exp(m * x + c);
//   noise = randn(size(x)) * 0.2;
//   y_observed = y + noise;
//   data = [x', y_observed'];

int kNumObservations;
vector<double> data;

void GenData(int N) {
  kNumObservations = N;
  data.resize(N*2);
  double step = 5.0 / N;
  const double m = 0.3, c = 0.1;
  for(int i=0;i<N;++i) {
    double x = i * step;
    double y = exp(m * x + c);
    double noise = rand() / (double)RAND_MAX * 0.2;
    data[i*2] = x;
    data[i*2+1] = y;
  }
}

const int N = 2;

struct ExponentialResidual {
  ExponentialResidual(double x, double y, int i)
      : x_(x), y_(y), idx(i) {}
  template <typename T> bool operator()(T const* const* params,
                                        T* residual) const {
    const T* m = params[0];
    const T* c = &(params[0][N]);
    int i = idx % N;
    int prev = i-1;
    if(prev < 0) prev += N;
    int next = i+1;
    if(next >= N) next -= N;

    residual[0] = T(y_) - exp(
      (m[prev] + m[i] + m[next]) * T(x_) +
      (c[prev] + c[i] + c[next])
    );
    return true;
  }
 private:
  const double x_;
  const double y_;
  int idx;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  double* m = new double[N*2];
  for(int i=0;i<N*2;++i) m[i] = 0;

  GenData(10000000);
  // 50000 * 4,000,000 observations
  // 10000 * 47 * 9 = 4,500,000 prior terms
  // 10000 * 47 * 3 = 1,500,000 parameters

  Problem problem;
  for (int i = 0; i < kNumObservations; ++i) {
    auto* cost_fun = new DynamicAutoDiffCostFunction<ExponentialResidual, 4>(
        new ExponentialResidual(data[2 * i], data[2 * i + 1], i));
    cost_fun->AddParameterBlock(N*2);
    cost_fun->SetNumResiduals(1);
    problem.AddResidualBlock(cost_fun, NULL, m);
  }
  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  //std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
  //std::cout << "Final   m: " << m << " c: " << c << "\n";
  return 0;
}
