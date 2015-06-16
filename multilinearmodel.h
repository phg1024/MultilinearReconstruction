#ifndef MULTILINEARMODEL_H
#define MULTILINEARMODEL_H

#include "tensor.hpp"


class MultilinearModel
{
public:
  MultilinearModel();

  MultilinearModel project(const vector<int> &indices);

private:
  Tensor3 core;
  Tensor2 u0, u1;     // unfolded tensor in 0, 1 dimension
};

#endif // MULTILINEARMODEL_H
