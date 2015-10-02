#ifndef BASICMESH_H
#define BASICMESH_H

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class BasicMesh
{
public:
  BasicMesh() {}
  BasicMesh(const string& filename);

  Vector3d vertex(int i) const {
    return verts.row(i);
  }
  Vector3i face(int i) const {
    return faces.row(i);
  }
  Vector3d normal(int i) const {
    return norms.row(i);
  }

  int NumVertices() const { return static_cast<int>(verts.rows()); }
  int NumFaces() const { return static_cast<int>(faces.rows()); }

  bool LoadOBJMesh(const string &filename);
  void ComputeNormals();
  void UpdateVertices(const VectorXd& vertices);

private:
  MatrixX3d verts;
  MatrixX3i faces;
  // Per-vertex normal vector
  MatrixX3d norms;
};

#endif // BASICMESH_H
