#include <iostream>
#include "../basicmesh.h"
using namespace std;

int main(int argc, char** argv) {
  if(argc < 3) {
    cout << "Usage: " << argv[0] << " inputmesh outputmesh" << endl;
    return 0;
  } else {
    BasicMesh mesh(argv[1]);
    mesh.BuildHalfEdgeMesh();
    mesh.Subdivide();
    mesh.Write(argv[2]);
    return 0;
  }
}
