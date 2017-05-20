#include <QtCore/QCoreApplication>

#include "phgutils.h"
#include "Geometry/MeshLoader.h"
#include "Geometry/MeshWriter.h"
#include "Geometry/Mesh.h"

int main(int argc, char *argv[])
{
    if(argc < 3) {
      cout << "Usage: " << argv[0] << " base_mesh partial_mesh" << endl;
      return 1;
    }
    PhGUtils::OBJLoader loader1, loader2;
    loader1.load(argv[1]);
    loader2.load(argv[2]);

    PhGUtils::QuadMesh m1;
    PhGUtils::TriMesh m2;
    m1.initWithLoader(loader1);
    m2.initWithLoader(loader2);

    // for each face in the quad mesh, consider it interesting if it is found in the tri mesh
    int nFaces = m1.faceCount();
    int nFaces_tri = m2.faceCount();
    const float THRES = 1e-5;

    vector<int> goodfaces;
    for (int i = 0; i < nFaces; ++i) {
      bool matched = false;
      PhGUtils::QuadMesh::face_t &f = m1.face(i);
      PhGUtils::QuadMesh::vert_t vi[4];
      vi[0] = m1.vertex(f.x);
      vi[1] = m1.vertex(f.y);
      vi[2] = m1.vertex(f.z);
      vi[3] = m1.vertex(f.w);

      for (int j = 0; j < nFaces_tri; ++j) {
        PhGUtils::TriMesh::face_t &fj = m2.face(j);
        PhGUtils::TriMesh::vert_t vj[3];

        vj[0] = m2.vertex(fj.x);
        vj[1] = m2.vertex(fj.y);
        vj[2] = m2.vertex(fj.z);

        for (int k = 0; k < 4; ++k) {
          for (int l = 0; l < 3; ++l) {
            if (vi[k].distanceTo(vj[l]) < THRES) {
              matched = true;
              goto matchfound;
            }
          }
        }
      }
matchfound:
      if (matched) goodfaces.push_back(i);
    }

    PhGUtils::OBJWriter writer;
    writer.save(m1, goodfaces, "cutted.obj");

    ofstream fout("indices.txt");
    for (auto x : goodfaces) {
      fout << x << endl;
    }
    fout.close();

    return 0;
}
