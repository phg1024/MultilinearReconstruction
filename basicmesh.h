#ifndef BASICMESH_H
#define BASICMESH_H

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include "common.h"

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
  Vector3i face_texture(int i) const {
    return face_tex_index.row(i);
  }
  Vector3d normal(int i) const {
    return norms.row(i);
  }
  Vector3d vertex_normal(int i) const {
    return vertex_norms.row(i);
  }
  Vector2d texture_coords(int i) const {
    return texcoords.row(i);
  }

  int NumVertices() const { return static_cast<int>(verts.rows()); }
  int NumFaces() const { return static_cast<int>(faces.rows()); }

  bool LoadOBJMesh(const string &filename);
  void ComputeNormals();
  void UpdateVertices(const VectorXd& vertices);

  vector<int> GetNeighbors() const;

private:
  unordered_map<int, int> vert_face_map;

  MatrixX3d verts;
  MatrixX3i faces, face_tex_index;
  MatrixX2d texcoords;

  // Per-face normal vector
  MatrixX3d norms;
  MatrixX3d vertex_norms;
};

#endif // BASICMESH_H
