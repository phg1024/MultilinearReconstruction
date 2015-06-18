#ifndef MULTILINEARRECONSTRUCTOR_HPP
#define MULTILINEARRECONSTRUCTOR_HPP

#include "common.h"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

using namespace std;
using namespace Eigen;

#include "multilinearmodel.h"
#include "constraints.h"
#include "utils.hpp"

struct CameraParameters {
  double fx, fy, cx, cy;
};

struct ModelParameters {
  static const int nFACSDim = 47;
  VectorXd Wid;               // identity weights
  VectorXd Wexp, Wexp_FACS;   // expression weights
  Quaterniond R;              // rotation
  Vector3d T;                 // translation
};

template <typename Constraint>
struct ReconstructionParameters {
  int imageWidth, imageHeight;
  vector<Constraint> cons;
};

struct MultilinearModelPrior {
  VectorXd Wid_avg, Wexp_avg;
  VectorXd Wid0, Wexp0;       // identity and expression prior
  MatrixXd Uid, Uexp;
  MatrixXd sigma_Wid, sigma_Wexp;
  MatrixXd inv_sigma_Wid, inv_sigma_Wexp;
  double weight_Wid, weight_Wexp;

  void load(const string &filename_id, const string &filename_exp) {
    cout << "loading prior data ..." << endl;
    const string fnwid = filename_id;
    ifstream fwid(fnwid, ios::in | ios::binary);

    int ndims;
    fwid.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    cout << "identity prior dim = " << ndims << endl;

    Wid_avg.resize(ndims);
    Wid0.resize(ndims);
    sigma_Wid.resize(ndims, ndims);

    fwid.read(reinterpret_cast<char*>(Wid_avg.data()), sizeof(double)*ndims);
    fwid.read(reinterpret_cast<char*>(Wid0.data()), sizeof(double)*ndims);
    fwid.read(reinterpret_cast<char*>(sigma_Wid.data()), sizeof(double)*ndims*ndims);

    int m, n;
    fwid.read(reinterpret_cast<char*>(&m), sizeof(int));
    fwid.read(reinterpret_cast<char*>(&n), sizeof(int));
    cout << "Uid size: " << m << 'x' << n << endl;
    Uid.resize(m, n);
    fwid.read(reinterpret_cast<char*>(Uid.data()), sizeof(double)*m*n);

    fwid.close();

    message("identity prior loaded.");
    /*
    cout << "Wid_avg = " << Wid_avg << endl;
    cout << "Wid0 = " << Wid0 << endl;
    cout << "sigma_Wid = " << sigma_Wid << endl;
    cout << "Uid = " << Uid << endl;
    */

    message("processing identity prior.");
    inv_sigma_Wid = sigma_Wid.inverse();
    message("done");

    const string fnwexp = filename_exp;
    ifstream fwexp(fnwexp, ios::in | ios::binary);

    fwexp.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    cout << "expression prior dim = " << ndims << endl;

    Wexp0.resize(ndims);
    Wexp_avg.resize(ndims);
    sigma_Wexp.resize(ndims, ndims);

    fwexp.read(reinterpret_cast<char*>(Wexp_avg.data()), sizeof(double)*ndims);
    fwexp.read(reinterpret_cast<char*>(Wexp0.data()), sizeof(double)*ndims);
    fwexp.read(reinterpret_cast<char*>(sigma_Wexp.data()), sizeof(double)*ndims*ndims);

    fwexp.read(reinterpret_cast<char*>(&m), sizeof(int));
    fwexp.read(reinterpret_cast<char*>(&n), sizeof(int));
    cout << "Uexp size: " << m << 'x' << n << endl;
    Uexp.resize(m, n);
    fwexp.read(reinterpret_cast<char*>(Uexp.data()), sizeof(double)*m*n);

    fwexp.close();

    message("expression prior loaded.");
    /*
    cout << "Wexp_avg = " << Wexp_avg << endl;
    cout << "Wexp0 = " << Wexp0 << endl;
    cout << "sigma_Wexp = " << sigma_Wexp << endl;
    cout << "Uexp = " << Uexp << endl;
    */
    message("processing expression prior.");
    inv_sigma_Wexp = sigma_Wexp.inverse();
    message("done.");
  }
};

struct OptimizationParameters {
  int maxIters;
  double errorThreshold;
  double errorDiffThreshold;
};

template <typename Constraint>
class SingleImageReconstructor {
public:
  SingleImageReconstructor(){}
  void loadModel(const string &filename);
  void loadPriors(const string &filename_id, const string &filename_exp);
  void setIndices(const vector<int> &indices_vec) { indices = indices_vec; }

  void setConstraints(const vector<Constraint> &cons) { params_recon.cons = cons; }
  void setImageSize(int w, int h) {
    params_recon.imageWidth = w;
    params_recon.imageHeight = h;
  }
  void setOptimizationParameters(const OptimizationParameters &params) {
    params_opt = params;
  }

  bool reconstruct();

  const VectorXd &getIdentityWeights() const { return params_model.Wid; }
  const VectorXd &getExpressionWeights() const { return params_model.Wexp_FACS; }

private:
  MultilinearModel model, model_projected;
  vector<int> indices;
  MultilinearModelPrior prior;

  CameraParameters params_cam;
  ModelParameters params_model;
  ReconstructionParameters<Constraint> params_recon;
  OptimizationParameters params_opt;
};

template <typename Constraint>
void SingleImageReconstructor<Constraint>::loadModel(const string &filename)
{
  model = MultilinearModel(filename);
}

template <typename Constraint>
void SingleImageReconstructor<Constraint>::loadPriors(const string &filename_id, const string &filename_exp)
{
  prior.load(filename_id, filename_exp);
}

template <typename Constraint>
bool SingleImageReconstructor<Constraint>::reconstruct()
{
  return true;
}

#endif // MULTILINEARRECONSTRUCTOR_HPP


