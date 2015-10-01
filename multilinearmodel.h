#ifndef MULTILINEARMODEL_H
#define MULTILINEARMODEL_H

#include "tensor.hpp"

class MultilinearModel
{
public:
  MultilinearModel(){}
  explicit MultilinearModel(const string &filename);

  MultilinearModel project(const vector<int> &indices) const;

  void UpdateTM0(const Tensor1 &w);
  void UpdateTM1(const Tensor1 &w);
  void UpdateTMWithTM0(const Tensor1 &w);
  void UpdateTMWithTM1(const Tensor1 &w);
  void ApplyWeights(const Tensor1 &w0, const Tensor1 &w1);

  const Tensor1& GetTM() const {
    return tm;
  }
private:
  void UnfoldCoreTensor();

private:
  Tensor3 core;
  Tensor2 tu0, tu1;     // unfolded tensor in 0, 1 dimension

  Tensor2 tm0, tm1;  // tensor after mode product
  Tensor1 tm;        // tensor after 2 mode product
};

#endif // MULTILINEARMODEL_H
