#ifndef MULTILINEARMODELBUILDER_H
#define MULTILINEARMODELBUILDER_H

#include "blendshape_data.h"
#include "tensor.hpp"
#include "utils.hpp"

class MultilinearModelBuilder {
public:
  MultilinearModelBuilder(){}
  void build(){
    cout << "building multilinear model ..." << endl;

    vector<BlendShape> shapes;

    const int nShapes = 150;			// 150 identity
    const int nExprs = 47;				// 46 expressions + 1 neutral
    const int nVerts = 11510;			// 11510 vertices for each mesh

    const string path = "/home/phg/Data/FaceWarehouse_Data_0/";
    const string foldername = "Tester_";
    const string bsfolder = "Blendshape";
    const string filename = "shape.bs";

    shapes.resize(nShapes);
    for(int i=0;i<nShapes;i++) {
      stringstream ss;
      ss << path << foldername << (i+1) << "/" << bsfolder + "/" + filename;

      shapes[i].read(ss.str());
    }
    int nCoords = nVerts * 3;

    // create an order 3 tensor for the blend shapes
    Tensor3 t(nShapes, nExprs, nCoords);

    // fill in the data
    for(int i=0;i<shapes.size();i++) {
      const BlendShape& bsi = shapes[i];
      for(int j=0;j<bsi.expressionCount();j++) {
        const BlendShape::shape_t& bsij = bsi.expression(j);

        for(int k=0, cidx = 0;k<nVerts;k++, cidx+=3) {
          const BlendShape::vert_t& v = bsij[k];

          t(i, j, cidx) = v.x;
          t(i, j, cidx+1) = v.y;
          t(i, j, cidx+2) = v.z;
        }
      }
    }

    cout << "Tensor assembled." << endl;

    // create deformation map
    Tensor2 distmap(nShapes, nVerts);
    for(int i=0;i<shapes.size();i++) {
      const BlendShape::shape_t& bsi0 = shapes[i].expression(0);
      const BlendShape& bsi = shapes[i];
      for(int j=1;j<bsi.expressionCount();j++) {
        const BlendShape::shape_t& bsij = bsi.expression(j);

        for(int k=0, cidx = 0;k<nVerts;k++, cidx+=3) {
          const BlendShape::vert_t& v0 = bsi0[k];
          const BlendShape::vert_t& v = bsij[k];
          float dx = v.x - v0.x, dy = v.y - v0.y, dz = v.z - v0.z;
          distmap(i, k) += sqrt(dx*dx+dy*dy+dz*dz);
        }
      }
    }

    distmap.write("distmap.txt");

    // perform svd to get core tensor
    cout << "Performing SVD on the blendshapes ..." << endl;
    int ms[2] = {0, 1};		// only the first two modes
    int ds[2] = {50, 25};	// pick 50 for identity and 25 for expression
    vector<int> modes(ms, ms+2);
    vector<int> dims(ds, ds+2);
    auto comp2 = t.svd(modes, dims);
    cout << "SVD done." << endl;

    auto tcore = std::get<0>(comp2);
    auto tus = std::get<1>(comp2);
    cout << "writing core tensor ..." << endl;
    tcore.write("blendshape_core.tensor");
    cout << "writing U tensors ..." << endl;
    for(int i=0;i<tus.size();i++) {
      tus[i].write("blendshape_u_" + toString(ms[i]) + ".tensor");
    }

    cout << "Validation begins ..." << endl;
    Tensor3 tin;
    tin.read("blendshape_core.tensor");

    cout << "Core tensor dimensions = "
      << tin.layers() << "x"
      << tin.rows() << "x"
      << tin.cols() << endl;

    double maxDiffio = 0;

    for(int i=0;i<tin.layers();i++) {
      for(int j=0;j<tin.rows();j++) {
        for(int k=0;k<tin.cols();k++) {
          maxDiffio = std::max(fabs(tin(i, j, k) - tcore(i, j, k)), maxDiffio);
        }
      }
    }
    cout << "Max difference io = " << maxDiffio << endl;

    tin = tin.modeProduct(tus[0], 0).modeProduct(tus[1], 1);

    cout << "Dimensions = "
      << tin.layers() << "x"
      << tin.rows() << "x"
      << tin.cols() << endl;
    double maxDiff = 0;

    for(int i=0;i<tin.layers();i++) {
      for(int j=0;j<tin.rows();j++) {
        for(int k=0;k<tin.cols();k++) {
          maxDiff = std::max(fabs(tin(i, j, k) - t(i, j, k)), maxDiff);
        }
      }
    }

    cout << "Max difference = " << maxDiff << endl;
    cout << "done" << endl;

  }
};

#endif // MULTILINEARMODELBUILDER_H

