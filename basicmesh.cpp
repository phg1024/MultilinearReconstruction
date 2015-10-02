#include "basicmesh.h"
#include "Geometry/MeshLoader.h"

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
  verts.resize(V.size(), 3);
  for (int i = 0; i < nverts; ++i) {
    verts(i, 0) = V[i].x;
    verts(i, 1) = V[i].y;
    verts(i, 2) = V[i].z;
  }

  int nfaces = 0;
  auto F = loader.getFaces();
  for (int i = 0; i < F.size(); ++i) {
    nfaces += F[i].v.size()-2;
  }

  faces.resize(nfaces, 3);
  // triangulate the mesh
  for (int i = 0, faceidx = 0; i < F.size(); ++i) {
    for (int j = 1; j < F[i].v.size()-1; ++j, ++faceidx) {
      faces.row(faceidx) = Vector3i(F[i].v[0], F[i].v[j], F[i].v[j+1]);
    }
  }

  cout << filename << " loaded." << endl;
  cout << nfaces << " faces." << endl;
  cout << V.size() << " vertices." << endl;
  return true;
}

void BasicMesh::ComputeNormals()
{
  norms.resize(faces.rows(), 3);

#pragma omp parallel for
  for(int i=0;i<faces.rows();++i) {
    auto v0 = Vector3d(verts.row(faces(i, 0)));
    auto v1 = Vector3d(verts.row(faces(i, 1)));
    auto v2 = Vector3d(verts.row(faces(i, 2)));

    auto v0v1 = v1 - v0;
    auto v0v2 = v2 - v0;
    auto n = v0v1.cross(v0v2);
    n.normalize();

    norms.row(i) = n;
  }
  cout << "Normals computed." << endl;
}
