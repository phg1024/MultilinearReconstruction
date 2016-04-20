#ifndef BASICMESH_H
#define BASICMESH_H

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include "common.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  HalfEdgeMesh;

using namespace std;
using namespace Eigen;

class BasicMesh
{
public:
  BasicMesh() {}
  BasicMesh(const string& filename);

  void set_vertex(int i, const Vector3d& v) {
    verts.row(i) = v;
  }
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

  const MatrixX3d& vertices() const {
    return verts;
  }
  MatrixX3d& vertices() {
    return verts;
  }

  int NumVertices() const { return static_cast<int>(verts.rows()); }
  int NumFaces() const { return static_cast<int>(faces.rows()); }

  bool LoadOBJMesh(const string &filename);
  void ComputeNormals();
  void UpdateVertices(const VectorXd& vertices);
  void Subdivide();

  vector<int> GetNeighbors() const;

  void Write(const string& filename) const;

  template <typename Pred>
  vector<int> filterFaces(Pred p) {
    vector<int> v;
    for(int i=0;i<NumFaces();++i) {
      if( p(faces.row(i)) ) {
        v.push_back(i);
      }
    }
    return v;
  }
  template <typename Pred>
  vector<int> filterVertices(Pred p) {
    vector<int> v;
    for(int i=0;i<NumVertices();++i) {
      if( p(verts.row(i)) ) {
        v.push_back(i);
      }
    }
    return v;
  }
  MatrixX3d samplePoints(int points_per_face, double zcutoff) const {
    int npoints = 0;
    vector<int> validfaces;

    for (int i = 0; i < NumFaces(); ++i) {
      // sample 8 points per face
      int v1 = faces(i, 0), v2 = faces(i, 1), v3 = faces(i, 2);
      double z1 = verts(v1, 2), z2 = verts(v2, 2), z3 = verts(v3, 2);
      double zc = (z1 + z2 + z3) / 3.0;
      if (zc > zcutoff) {
        npoints += points_per_face;
        validfaces.push_back(i);
      }
    }
    cout << "npoints = " << npoints << endl;
    MatrixXd P(npoints, 3);
    for (size_t i = 0, offset=0; i < validfaces.size(); ++i) {
      int fidx = validfaces[i];
      int v1 = faces(fidx, 0), v2 = faces(fidx, 1), v3 = faces(fidx, 2);
      double x1 = verts(v1, 0), x2 = verts(v2, 0), x3 = verts(v3, 0);
      double y1 = verts(v1, 1), y2 = verts(v2, 1), y3 = verts(v3, 1);
      double z1 = verts(v1, 2), z2 = verts(v2, 2), z3 = verts(v3, 2);

      for(int j=0;j<points_per_face;++j) {
        // sample a point
        double alpha = rand()/(double)RAND_MAX,
          beta = rand()/(double)RAND_MAX * (1-alpha),
          gamma = 1.0 - alpha - beta;

        double xij = x1*alpha + x2*beta + x3*gamma;
        double yij = y1*alpha + y2*beta + y3*gamma;
        double zij = z1*alpha + z2*beta + z3*gamma;
        P.row(offset) = Vector3d(xij, yij, zij); ++offset;
      }
    }
    cout << "points sampled." << endl;
    return P;
  }

  void BuildHalfEdgeMesh();

private:
  unordered_map<int, int> vert_face_map;

  MatrixX3d verts;
  MatrixX3i faces, face_tex_index;
  MatrixX2d texcoords;

  // Per-face normal vector
  MatrixX3d norms;
  MatrixX3d vertex_norms;

  template <typename Handle>
  struct HandleHasher {
    std::size_t operator()(const Handle& h) const {
      return h.idx();
    }
  };

  // This mesh stored in half edge data structure
  // Makes mesh manipulation easier
  vector<HalfEdgeMesh::VertexHandle> vhandles;
  unordered_map<HalfEdgeMesh::VertexHandle, int, HandleHasher<HalfEdgeMesh::VertexHandle>> vhandles_map;
  vector<HalfEdgeMesh::FaceHandle> fhandles;
  unordered_map<HalfEdgeMesh::FaceHandle, int, HandleHasher<HalfEdgeMesh::FaceHandle>> fhandles_map;
  HalfEdgeMesh hemesh;
};

#endif // BASICMESH_H
