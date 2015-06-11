#ifndef BLENDSHAPE_DATA_H
#define BLENDSHAPE_DATA_H

#include "common.h"
#include "utils.hpp"

class BlendShape
{
public:
  struct vert_t {
    float x, y, z;
  };
  typedef vector<vert_t> shape_t;

  BlendShape(void);
  ~BlendShape(void);

  const shape_t& expression(int idx) const {
    return exprList[idx];
  }

  int expressionCount() const {
    return nShapes;
  }

public:
  bool read(const string& filename);

private:
  int nVerts;
  int nShapes;
  int nFaces;

  vector<shape_t> exprList;
};

BlendShape::BlendShape(void)
{
}


BlendShape::~BlendShape(void)
{
}

bool BlendShape::read(const string& filename) {
  message("reading blendshape file " + filename);
  ifstream fin;
  fin.open(filename, ios::in | ios::binary );

  if( !fin ) {
    error("Failed to read file " + filename);
    return false;
  }

  fin.read( reinterpret_cast<char*>(&nShapes), sizeof(int) );			// nShape = 46
  fin.read( reinterpret_cast<char*>(&nVerts), sizeof(int) );			// nVerts = 11510
  fin.read( reinterpret_cast<char*>(&nFaces), sizeof(int) );			// nFaces = 11540

  nShapes++;	// plus the neutral shape

  // Load neutral expression B_0
  exprList.resize(nShapes);
  // Load other expressions B_i ( 1 <= i <= 46 )
  for( int exprId=0; exprId<nShapes; exprId++ ){
    shape_t& expr = exprList[ exprId ];
    expr.resize( nVerts );
    fin.read( reinterpret_cast<char*>(&expr[0]), sizeof(vert_t) * nVerts );
  }

  message("done.");
  fin.close();
  return true;
}
#endif // BLENDSHAPE_DATA_H

