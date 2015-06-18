#ifndef MULTILINEARMODEL_H
#define MULTILINEARMODEL_H

#include "tensor.hpp"

class MultilinearModel
{
public:
  MultilinearModel(){}
  explicit MultilinearModel(const string &filename);

  MultilinearModel project(const vector<int> &indices);

  void updateTM0(const Tensor1 &w);
  void updateTM1(const Tensor1 &w);
  void updateTMWithTM0(const Tensor1 &w);
  void updateTMWithTM1(const Tensor1 &w);
  void applyWeights(const Tensor1 &w0, const Tensor1 &w1);

private:
  void unfold();

private:
  Tensor3 core;
  Tensor2 tu0, tu1;     // unfolded tensor in 0, 1 dimension

  Tensor2 tm0, tm1;  // tensor after mode product
  Tensor1 tm;        // tensor after 2 mode product
};

#endif // MULTILINEARMODEL_H
