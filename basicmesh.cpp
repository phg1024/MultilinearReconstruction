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
  bool has_texcoords = T.size() > 0;
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
      if(has_texcoords) {
        face_tex_index.row(faceidx) = Vector3i(F[i].t[0], F[i].t[j], F[i].t[j+1]);
      }
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

void BasicMesh::Subdivide() {
  // For each edge, compute its center point
  struct edge_t {
    edge_t() {}
    edge_t(int s, int t) : s(s), t(t) {}
    bool operator<(const edge_t& other) const {
      if(s < other.s) return true;
      else if( s > other.s ) return false;
      else return t < other.t;
    }
    int s, t;
  };
  map<edge_t, Vector3d> midpoints;

  const int num_faces = NumFaces();
  for(int i=0;i<num_faces;++i) {
    auto vidx0 = faces(i, 0);
    auto vidx1 = faces(i, 1);
    auto vidx2 = faces(i, 2);

    auto v0 = Vector3d(verts.row(vidx0));
    auto v1 = Vector3d(verts.row(vidx1));
    auto v2 = Vector3d(verts.row(vidx2));

    if(midpoints.count(edge_t(vidx0, vidx1)) == 0
    && midpoints.count(edge_t(vidx1, vidx0)) == 0)
      midpoints.insert(make_pair(edge_t(vidx0, vidx1), 0.5 * (v0 + v1)));

    if(midpoints.count(edge_t(vidx1, vidx2)) == 0
    && midpoints.count(edge_t(vidx2, vidx1)) == 0)
      midpoints.insert(make_pair(edge_t(vidx1, vidx2), 0.5 * (v1 + v2)));

    if(midpoints.count(edge_t(vidx2, vidx0)) == 0
    && midpoints.count(edge_t(vidx0, vidx2)) == 0)
      midpoints.insert(make_pair(edge_t(vidx2, vidx0), 0.5 * (v2 + v0)));
  }

  // Now create a new set of vertices and faces
  const int num_verts = NumVertices() + midpoints.size();
  MatrixX3d new_verts(num_verts, 3);
  new_verts.topRows(NumVertices()) = verts;
  map<edge_t, int> midpoints_indices;
  int new_idx = NumVertices();
  for(auto p : midpoints) {
    midpoints_indices.insert(make_pair(p.first, new_idx));
    midpoints_indices.insert(make_pair(edge_t(p.first.t, p.first.s), new_idx));

    new_verts.row(new_idx) = p.second;
    ++new_idx;
  }

  MatrixX3i new_faces(num_faces*4, 3);
  for(int i=0;i<num_faces;++i) {
    auto vidx0 = faces(i, 0);
    auto vidx1 = faces(i, 1);
    auto vidx2 = faces(i, 2);

    int nvidx01 = midpoints_indices[edge_t(vidx0, vidx1)];
    int nvidx12 = midpoints_indices[edge_t(vidx1, vidx2)];
    int nvidx20 = midpoints_indices[edge_t(vidx2, vidx0)];

    // add the 4 new faces
    new_faces.row(i*4+0) = Vector3i(vidx0, nvidx01, nvidx20);
    new_faces.row(i*4+1) = Vector3i(nvidx20, nvidx01, nvidx12);
    new_faces.row(i*4+2) = Vector3i(nvidx20, nvidx12, vidx2);
    new_faces.row(i*4+3) = Vector3i(nvidx01, vidx1, nvidx12);
  }

  verts = new_verts;
  faces = new_faces;

  // Update the normals after subdivision
  ComputeNormals();
}

void BasicMesh::Write(const string &filename) {
  string content;
  // write verts
  for (int i = 0,offset=0; i < NumVertices(); ++i) {
    content += "v ";
    content += to_string(verts(i, 0)) + " "; ++offset;
    content += to_string(verts(i, 1)) + " "; ++offset;
    content += to_string(verts(i, 2)) + "\n"; ++offset;
  }

  // write faces
  for (int i = 0, offset = 0; i < NumFaces(); ++i) {
    content += "f ";
    content += to_string(faces(i, 0) + 1) + " "; ++offset;
    content += to_string(faces(i, 1) + 1) + " "; ++offset;
    content += to_string(faces(i, 2) + 1) + "\n"; ++offset;
  }

  ofstream fout(filename);
  fout << content << endl;
  fout.close();
}
