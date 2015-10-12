#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "common.h"

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

using Tensor1 = VectorXd;

class Tensor2 {
public:
  Tensor2(){}
  Tensor2(int m, int n):data(MatrixXd(m, n)){}
  Tensor2(const MatrixXd &M):data(M){}

  Tensor2(std::initializer_list<std::initializer_list<double>> l) {
    int m = l.size();
    int n = l.begin()->size();
    resize(m, n);
    auto li = l.begin();
    for(int i=0;i<m;++i) {
      auto lij = li->begin();
      assert(li->size() == n);
      for(int j=0;j<n;++j) {
        data(i, j) = *lij;
        ++lij;
      }
      ++li;
    }
  }

  void resize(int m, int n) {
    data.resize(m, n);
  }

  int rows() const { return data.rows(); }
  int cols() const { return data.cols(); }

  double& operator()(int i, int j) { return data(i, j); }
  const double& operator() (int i, int j) const { return data(i, j); }

  MatrixXd::RowXpr row(int i) { return data.row(i); }
  MatrixXd::ConstRowXpr row(int i) const { return data.row(i); }

  MatrixXd::ColXpr col(int i) { return data.col(i); }
  MatrixXd::ConstColXpr col(int i) const { return data.col(i); }

  VectorXd operator*(const VectorXd &v) const { return data * v; }

  static Tensor2 FoldByColumn(Tensor1& v, int nrows, int ncols) {
    Tensor2 t;
    t.data = Eigen::Map<MatrixXd>(v.data(), ncols, nrows).transpose();
    return t;
  }

  static Tensor2 FoldByRow(Tensor1& v, int nrows, int ncols) {
    Tensor2 t;
    t.data = Eigen::Map<MatrixXd>(v.data(), nrows, ncols);
    return t;
  }

  void Unfold(Tensor1 &v) const {
    v.resize(rows()*cols());
    for(int i=0,idx=0;i<rows();++i) {
      for(int j=0;j<cols();++j,++idx) {
        v(idx) = this->operator()(i, j);
      }
    }
  }

  Tensor1 Unfold() const {
    Tensor1 v;
    Unfold(v);
    return v;
  }

  template <int Mode> void ModeProduct(const Tensor1 &v, Tensor1 &u) {}

  template <int Mode>
  Tensor1 ModeProduct(const Tensor1 &v) {
    Tensor1 u(v.size());
    ModeProduct<Mode>(v, u);
    return u;
  }

  JacobiSVD<MatrixXd> svd_econ() const {
    return data.jacobiSvd(ComputeFullU | ComputeThinV);
  }

  double norm() const {
    return data.norm();
  }

  void Print(const string& title = "", ostream& os = cout) const {
    if( !title.empty() ) cout << title << " = " << endl;;
    os << (*this) << endl;
  }

  bool Read(const string& filename) {
    try {
      cout << "reading tensor to file " << filename << endl;
      fstream fin;
      fin.open(filename, ios::in | ios::binary);

      int m, n;

      fin.read(reinterpret_cast<char*>(&(m)), sizeof(int));
      fin.read(reinterpret_cast<char*>(&(n)), sizeof(int));
      cout << "tensor size = " << m << "x" << n << endl;
      data.resize(m, n);
      fin.read(reinterpret_cast<char*>(data.data()), sizeof(double)*m*n);

      fin.close();

      cout << "done." << endl;
      return true;
    }
    catch(...) {
      cerr << "Failed to write tensor to file " << filename << endl;
      return false;
    }
  }

  bool Write(const string& filename) {
    try {
      cout << "writing tensor to file " << filename << endl;
      int m = rows(), n = cols();

      fstream fout;
      fout.open(filename, ios::out | ios::binary);

      fout.write(reinterpret_cast<const char*>(&(m)), sizeof(int));
      fout.write(reinterpret_cast<const char*>(&(n)), sizeof(int));

      fout.write(reinterpret_cast<const char*>(data.data()), sizeof(double)*m*n);

      fout.close();

      cout << "done." << endl;
      return true;
    }
    catch(...) {
      cerr << "Failed to write tensor to file " << filename << endl;
      return false;
    }
  }

  const double* rawptr() const { return data.data(); }
  double* rawptr() { return data.data(); }

  friend Tensor2 operator+(const Tensor2& a, const Tensor2& b);
  friend Tensor2 operator-(const Tensor2& a, const Tensor2& b);
  friend bool operator==(const Tensor2& a, const Tensor2& b);
  friend bool operator!=(const Tensor2& a, const Tensor2& b);
  friend ostream& operator<<(ostream& os, const Tensor2& t);

private:
  MatrixXd data;
};

// u = (A^T * v)^T = (v^T * A)^T
template <>
inline void Tensor2::ModeProduct<0>(const Tensor1 &v, Tensor1 &u) {
  u = v.transpose() * data;
  u = u.transpose();
}

// u = A * v
template <>
inline void Tensor2::ModeProduct<1>(const Tensor1 &v, Tensor1 &u) {
  u = data * v;
}

inline Tensor2 operator+(const Tensor2& a, const Tensor2& b) {
  assert(a.rows() == b.rows());
  assert(a.cols() == b.cols());
  Tensor2 res(a.rows(), a.cols());
  res.data = a.data + b.data;
  return res;
}

inline Tensor2 operator-(const Tensor2& a, const Tensor2& b) {
  assert(a.rows() == b.rows());
  assert(a.cols() == b.cols());
  Tensor2 res(a.rows(), a.cols());
  res.data = a.data - b.data;
  return res;
}

inline bool operator==(const Tensor2& a, const Tensor2 &b) {
  if( a.rows() != b.rows() ) return false;
  if( a.cols() != b.cols() ) return false;
  for(int i=0;i<a.rows();++i) {
    for(int j=0;j<b.cols();++j) {
      if( a(i, j) != b(i, j) ) return false;
    }
  }
  return true;
}

inline bool operator!=(const Tensor2& a, const Tensor2& b) {
  return !(a == b);
}

inline ostream& operator<<(ostream& os, const Tensor2& t) {
  os << "{" << t.data << "}";
  return os;
}

class Tensor3 {
public:
  Tensor3(){}
  Tensor3(int l, int m, int n):data(vector<Tensor2>(l, Tensor2(m, n))){}
  Tensor3(std::initializer_list<std::initializer_list<std::initializer_list<double>>> l) {
    int num_layers = l.size();
    int num_rows = l.begin()->size();
    int num_cols = l.begin()->begin()->size();
    resize(num_layers, num_rows, num_cols);

    auto li = l.begin();
    for(int i=0;i<num_layers;++i) {
      assert(li->size() == num_rows);
      auto lij = li->begin();
      for(int j=0;j<num_rows;++j) {
        assert(lij->size() == num_cols);
        auto lijk = lij->begin();
        for(int k=0;k<num_cols;++k) {
          data[i](j, k) = *lijk;
          ++lijk;
        }
        ++lij;
      }
      ++li;
    }
  }

  void resize(int l, int m, int n) {
    data.resize(l, Tensor2(m, n));
  }

  int layers() const { return data.size(); }
  int rows() const {
    if( data.empty() ) return 0;
    else return data.front().rows();
  }
  int cols() const {
    if( data.empty() ) return 0;
    else return data.front().cols();
  }

  double &operator()(int i, int j, int k) { return data[i](j, k); }
  const double &operator()(int i, int j, int k) const { return data[i](j, k); }

  template<int Mode>
  void Unfold(Tensor2 &t) const{}

  template<int Mode>
  Tensor2 Unfold() const{
    Tensor2 t;
    Unfold<Mode>(t);
    return t;
  }

  Tensor2 Unfold(int mid) const {
    switch(mid) {
    case 0:
      return Unfold<0>();
    case 1:
      return Unfold<1>();
    case 2:
      return Unfold<2>();
    default:
      throw "Unsupported mode!";
    }
  }

  template <int Mode>
  static void Fold(const Tensor2 &A, int l, int m, int n, Tensor3 &t){}

  template <int Mode>
  static Tensor3 Fold(const Tensor2 &A, int l, int m, int n){
    Tensor3 t;
    Fold<Mode>(A, l, m, n, t);
    return t;
  }

  template <int Mode>
  void ModeProduct(const Tensor1 &v, Tensor2 &A) {}

  template <int Mode>
  Tensor2 ModeProduct(const Tensor1 &v) {
    Tensor2 A;
    ModeProduct<Mode>(v, A);
    return A;
  }

  template <int Mode>
  void ModeProduct(const Tensor2 &A, Tensor3 &t) {}

  template <int Mode>
  Tensor3 ModeProduct(const Tensor2 &A) {
    Tensor3 t;
    ModeProduct<Mode>(A, t);
    return t;
  }

  Tensor3 ModeProduct(const Tensor2 &A, int mid) {
    switch( mid ) {
    case 0:
      return ModeProduct<0>(A);
    case 1:
      return ModeProduct<1>(A);
    case 2:
      return ModeProduct<2>(A);
    default:
      throw "Unsupported mode!";
    }
  }

  tuple<Tensor3, vector<Tensor2>> svd(const vector<int> &modes,
                                       const vector<int> &dims) const {
    vector<MatrixXd> U, V;
    vector<VectorXd> s;

    for(auto mid : modes) {
      cout << "svd on mode " << mid << endl;

      // unfold in mode mid
      Tensor2 t2 = Unfold(mid);

      // compute svd
      MatrixXd Ui, Vi;
      VectorXd si;

      auto svd = t2.svd_econ();
      Ui = svd.matrixU();
      si = svd.singularValues();

      cout << "done." << endl;
      // store the svd results
      U.push_back(Ui); V.push_back(Vi); s.push_back(si);
    }

    // decompose the tensor, with truncation
    Tensor3 core = (*this);
    vector<Tensor2> tu;
    for(size_t i=0;i<modes.size();i++) {
      int mid = modes[i];
      MatrixXd u_truncated = U[i].block(0, 0, U[i].rows(), dims[i]);
      Tensor2 tui( u_truncated );
      Tensor2 tuit( u_truncated.transpose() );

      core = core.ModeProduct(tuit, mid);
      tu.push_back(tui);
    }

    return make_tuple(core, tu);
  }

  tuple<Tensor3, Tensor2, Tensor2, Tensor2> svd() const {
    vector<int> modes{0, 1, 2};
    vector<int> dims{layers(), rows(), cols()};
    auto res = svd(modes, dims);
    auto& Us = get<1>(res);
    return make_tuple(get<0>(res), Us[0], Us[1], Us[2]);
  }

  double norm() const {
    int num_elements = layers() * rows() * cols();
    return sqrt(std::accumulate(data.begin(), data.end(), 0.0,
                                [](double val, const Tensor2& t){
                                  double t_norm = t.norm();
                                  return val + t_norm * t_norm;
                                }));
  }

  void Print(const string& title="") const {
    if( !title.empty() )
      cout << title << " = " << endl;
    cout << (*this) << endl;
  }

  bool Read(const string& filename) {
    try {
      cout << "Reading tensor file " << filename << endl;
      fstream fin;
      fin.open(filename, ios::in | ios::binary);

      int l, m, n;
      fin.read(reinterpret_cast<char*>(&(l)), sizeof(int));
      fin.read(reinterpret_cast<char*>(&(m)), sizeof(int));
      fin.read(reinterpret_cast<char*>(&(n)), sizeof(int));

      this->resize(l, m, n);

      for(int i=0;i<l;i++) {
        Tensor2& ti = data[i];
        fin.read(reinterpret_cast<char*>(ti.rawptr()), sizeof(double)*m*n);
      }

      fin.close();

      cout << "done." << endl;

      return true;
    }
    catch( ... ) {
      cerr << "Failed to read tensor from file " << filename << endl;
      return false;
    }
  }

  bool Write(const string& filename) {
    try {
      cout << "writing tensor to file " << filename << endl;
      int l = layers(), m = rows(), n = cols();

      fstream fout;
      fout.open(filename, ios::out | ios::binary);

      fout.write(reinterpret_cast<char*>(&(l)), sizeof(int));
      fout.write(reinterpret_cast<char*>(&(m)), sizeof(int));
      fout.write(reinterpret_cast<char*>(&(n)), sizeof(int));

      for(int i=0;i<l;i++) {
        const Tensor2& ti = data[i];
        fout.write(reinterpret_cast<const char*>(ti.rawptr()), sizeof(double)*m*n);
      }

      fout.close();

      cout << "done." << endl;
      return true;
    }
    catch(...) {
      cerr << "Failed to write tensor to file " << filename << endl;
      return false;
    }
  }

  friend bool operator==(const Tensor3& a, const Tensor3& b);
  friend bool operator!=(const Tensor3& a, const Tensor3& b);

  friend Tensor3 operator+(const Tensor3& a, const Tensor3& b);
  friend Tensor3 operator-(const Tensor3& a, const Tensor3& b);
  friend ostream& operator<<(ostream& os, const Tensor3& t);

private:
  vector<Tensor2> data;
};

template<>
inline void Tensor3::Unfold<0>(Tensor2 &t) const {
  int l = layers(), m = rows(), n = cols();
  t.resize(l, m*n);
  #pragma omp parallel for
  for(int i=0;i<l;++i) {
    t.row(i) = data[i].Unfold();
  }
}

template<>
inline void Tensor3::Unfold<1>(Tensor2 &t) const {
  int l = layers(), m = rows(), n = cols();
  t.resize(m, l*n);
  #pragma omp parallel for
  for(int i=0;i<l;++i) {
    for(int j=0;j<m;++j) {
      for(int k=0, offset=i;k<n;++k, offset+=l) {
        t(j, offset) = data[i](j, k);
      }
    }
  }
}

template<>
inline void Tensor3::Unfold<2>(Tensor2 &t) const {
  int l = layers(), m = rows(), n = cols();
  t.resize(n, l*m);
  for(int i=0, offset=0;i<l;++i, offset+=m) {
    for(int j=0;j<m;++j) {
      int jidx = offset + j;
      for(int k=0;k<n;++k) {
        t(k, jidx) = data[i](j, k);
      }
    }
  }
}

template <>
inline void Tensor3::Fold<0>(const Tensor2 &A, int l, int m, int n, Tensor3 &t){
  t.resize(l, m, n);
  #pragma omp parallel for
  for(int i=0;i<l;++i) {
    for(int j=0, offset=0;j<m;++j,offset+=n) {
      for(int k=0;k<n;++k) {
        t(i, j, k) = A(i, k+offset);
      }
    }
  }
}

template <>
inline void Tensor3::Fold<1>(const Tensor2 &A, int l, int m, int n, Tensor3 &t){
  t.resize(l, m, n);
  #pragma omp parallel for
  for(int i=0;i<l;++i) {
    for(int j=0;j<m;++j) {
      for(int k=0, offset=0;k<n;++k,offset+=l) {
        t(i, j, k) = A(j, i+offset);
      }
    }
  }
}

template <>
inline void Tensor3::Fold<2>(const Tensor2 &A, int l, int m, int n, Tensor3 &t){
  t.resize(l, m, n);
  for(int i=0,offset=0;i<l;++i,offset+=m) {
    for(int j=0;j<m;++j) {
      for(int k=0;k<n;++k) {
        t(i, j, k) = A(k, j+offset);
      }
    }
  }
}

template <>
inline void Tensor3::ModeProduct<0>(const Tensor1 &v, Tensor2 &A) {
  int l = layers(), m = rows(), n = cols();
  A.resize(m, n);
  #pragma omp parallel for
  for(int i=0;i<m;++i) {
    for(int j=0;j<n;++j) {
      double val = 0;
      for(int k=0;k<l;++k) {
        val += data[k](i, j) * v(k);
      }
      A(i, j) = val;
    }
  }
}

template <>
inline void Tensor3::ModeProduct<1>(const Tensor1 &v, Tensor2 &A) {
  int l = layers(), m = rows(), n = cols();
  A.resize(l, n);
  #pragma omp parallel for
  for(int i=0;i<l;++i) {
    for(int j=0;j<n;++j) {
      double val = 0;
      for(int k=0;k<m;++k) {
        val += data[i](k, j) * v(k);
      }
      A(i, j) = val;
    }
  }
}

template <>
inline void Tensor3::ModeProduct<2>(const Tensor1 &v, Tensor2 &A) {
  int l = layers(), m = rows();
  A.resize(l, m);
  for(int i=0;i<l;++i) {
    A.row(i) = (data[i] * v).transpose();
  }
}

template <>
inline void Tensor3::ModeProduct<0>(const Tensor2 &A, Tensor3 &t) {
  int l = layers(), m = rows(), n = cols();
  assert(A.cols() == l); // size(A) = rows(A) x l
  Tensor2 tu = Unfold<0>();   // l x (m*n)
  Tensor2 t2(A.rows(), m*n);  // rows(A) x (m*n)
  #pragma omp parallel for
  for(int i=0;i<tu.cols();++i) {
    t2.col(i) = A * tu.col(i);  // [ rows(A) x l ] x l -> rows(A)
  }

  t = Fold<0>(t2, A.rows(), m, n);
}

template <>
inline void Tensor3::ModeProduct<1>(const Tensor2 &A, Tensor3 &t) {
  int l = layers(), m = rows(), n = cols();
  assert(A.cols() == m); // size(A) = rows(A) x m
  Tensor2 tu = Unfold<1>();
  Tensor2 t2(A.rows(), l*n);
  #pragma omp parallel for
  for(int i=0;i<tu.cols();++i) {
    t2.col(i) = A * tu.col(i);
  }
  t = Fold<1>(t2, l, A.rows(), n);
}

template <>
inline void Tensor3::ModeProduct<2>(const Tensor2 &A, Tensor3 &t) {
  int l = layers(), m = rows(), n = cols();
  assert(A.cols() == n);
  Tensor2 tu = Unfold<2>();
  Tensor2 t2(A.rows(), l*m);
  #pragma omp parallel for
  for(int i=0;i<tu.cols();++i) {
    t2.col(i) = A * tu.col(i);
  }
  t = Fold<2>(t2, l, m, A.rows());
}

inline bool operator==(const Tensor3& a, const Tensor3& b) {
  if( a.layers() != b.layers() ) return false;
  if( a.rows() != b.rows() ) return false;
  if( a.cols() != b.cols() ) return false;

  for(int i=0;i<a.layers();++i) {
    for(int j=0;j<a.rows();++j) {
      for(int k=0;k<a.cols();++k) {
        if( a(i, j, k) != b(i, j, k) ) return false;
      }
    }
  }
  return true;
}

inline Tensor3 operator+(const Tensor3& a, const Tensor3& b) {
  assert(a.layers() == b.layers());
  assert(a.rows() == b.rows());
  assert(a.cols() == b.cols());
  Tensor3 res(a.layers(), a.rows(), a.cols());
  for(int i=0;i<a.layers();++i) {
    res.data[i] = a.data[i] + b.data[i];
  }
  return res;
}

inline Tensor3 operator-(const Tensor3& a, const Tensor3& b) {
  assert(a.layers() == b.layers());
  assert(a.rows() == b.rows());
  assert(a.cols() == b.cols());
  Tensor3 res(a.layers(), a.rows(), a.cols());
  for(int i=0;i<a.layers();++i) {
    res.data[i] = a.data[i] - b.data[i];
  }
  return res;
}

inline bool operator!=(const Tensor3& a, const Tensor3& b) {
  return !(a == b);
}

inline ostream& operator<<(ostream& os, const Tensor3& t) {
  os << "{";
  for(int i=0;i<t.layers();i++) {
    os << t.data[i];
    if( i < t.layers() - 1 ) os << endl;
  }
  os << "}";
  return os;
}

#endif // TENSOR_HPP

