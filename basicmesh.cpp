#include <mutex>
#include "omp.h"
#include "basicmesh.h"
#include "Geometry/MeshLoader.h"

#include "boost/timer/timer.hpp"

/// @brief Load a mesh from an OBJ file
BasicMesh::BasicMesh(const string &filename)
{
  if(!LoadOBJMesh(filename)) {
    exit(-1);
  }
  ComputeNormals();
}

bool BasicMesh::LoadOBJMesh(const string& filename) {
  cout << "loading " << filename << endl;
  PhGUtils::OBJLoader loader;
  if(!loader.load(filename)) {
    cerr << "Failed to load mesh file " << filename << endl;
    return false;
  }

  // initialize the basic mesh
  auto V = loader.getVerts();

  int nverts = V.size();
  verts.resize(nverts, 3);
  for (int i = 0; i < nverts; ++i) {
    verts(i, 0) = V[i].x;
    verts(i, 1) = V[i].y;
    verts(i, 2) = V[i].z;
  }

  auto T = loader.getTexcoords();
  texcoords.resize(T.size(), 2);
  for(int i=0;i<T.size();++i) {
    texcoords(i, 0) = T[i].x;
    texcoords(i, 1) = T[i].y;
  }

  int nfaces = 0;
  auto F = loader.getFaces();
  for (int i = 0; i < F.size(); ++i) {
    nfaces += F[i].v.size()-2;
  }

  faces.resize(nfaces, 3);
  face_tex_index.resize(nfaces, 3);
  vert_face_map.clear();
  // triangulate the mesh
  for (int i = 0, faceidx = 0; i < F.size(); ++i) {
    for (int j = 1; j < F[i].v.size()-1; ++j, ++faceidx) {
      faces.row(faceidx) = Vector3i(F[i].v[0], F[i].v[j], F[i].v[j+1]);
      face_tex_index.row(faceidx) = Vector3i(F[i].t[0], F[i].t[j], F[i].t[j+1]);
    }
  }

  cout << filename << " loaded." << endl;
  cout << nfaces << " faces." << endl;
  cout << nverts << " vertices." << endl;
  return true;
}

void BasicMesh::ComputeNormals() {
  boost::timer::auto_cpu_timer timer("[BasicMesh] Normals computation time = %w seconds.\n");

  norms.resize(faces.rows(), 3);
  vertex_norms.resize(verts.size(), 3);
  vertex_norms.setZero();
  vector<double> area_sum(verts.size(), 0.0);

  omp_lock_t writelock;

  omp_init_lock(&writelock);
#pragma omp parallel for
  for(int i=0;i<faces.rows();++i) {
    auto vidx0 = faces(i, 0);
    auto vidx1 = faces(i, 1);
    auto vidx2 = faces(i, 2);

    auto v0 = Vector3d(verts.row(vidx0));
    auto v1 = Vector3d(verts.row(vidx1));
    auto v2 = Vector3d(verts.row(vidx2));

    auto v0v1 = v1 - v0;
    auto v0v2 = v2 - v0;
    auto n = v0v1.cross(v0v2);
    double area = n.norm();

    omp_set_lock(&writelock);
    vertex_norms.row(vidx0) += n;
    vertex_norms.row(vidx1) += n;
    vertex_norms.row(vidx2) += n;

    area_sum[vidx0] += area;
    area_sum[vidx1] += area;
    area_sum[vidx2] += area;
    omp_unset_lock(&writelock);

    n.normalize();
    norms.row(i) = n;
  }
  omp_destroy_lock(&writelock);

#pragma omp parallel for
  for(int i=0;i<vertex_norms.rows();++i) {
    vertex_norms.row(i) /= area_sum[i];
  }
}

void BasicMesh::UpdateVertices(const VectorXd &vertices) {
  boost::timer::auto_cpu_timer timer("[BasicMesh] Vertices update time = %w seconds.\n");
  const int num_vertices = NumVertices();
#pragma omp parallel for
  for(int i=0;i<num_vertices;++i) {
    const int offset = i * 3;
    verts(i, 0) = vertices(offset+0);
    verts(i, 1) = vertices(offset+1);
    verts(i, 2) = vertices(offset+2);
  }
}
