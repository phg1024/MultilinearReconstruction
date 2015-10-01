#include "multilinearmodel.h"

MultilinearModel::MultilinearModel(const string &filename)
{
  core.Read(filename);
  UnfoldCoreTensor();
}

MultilinearModel MultilinearModel::project(const vector<int> &indices) const
{
  //cout << "creating projected tensors..." << endl;
  // create a projected version of the model
  MultilinearModel newmodel;
  newmodel.core.resize(core.layers(), core.rows(), indices.size() * 3);

  for (int i = 0; i < core.layers(); i++) {
    for (int j = 0; j < core.rows(); j++) {
      for (int k = 0, idx = 0; k < indices.size(); k++, idx += 3) {
        int vidx = indices[k] * 3;
        newmodel.core(i, j, idx) = core(i, j, vidx);
        newmodel.core(i, j, idx + 1) = core(i, j, vidx + 1);
        newmodel.core(i, j, idx + 2) = core(i, j, vidx + 2);
      }
    }
  }

  newmodel.UnfoldCoreTensor();

  return newmodel;
}

void MultilinearModel::UpdateTM0(const Tensor1 &w)
{
  tm0 = core.ModeProduct<0>(w);
}

void MultilinearModel::UpdateTM1(const Tensor1 &w)
{
  tm1 = core.ModeProduct<1>(w);
}

void MultilinearModel::UpdateTMWithTM0(const Tensor1 &w)
{
  tm = tm0.ModeProduct<0>(w);
}

void MultilinearModel::UpdateTMWithTM1(const Tensor1 &w)
{
  tm = tm1.ModeProduct<0>(w);
}

void MultilinearModel::ApplyWeights(const Tensor1 &w0, const Tensor1 &w1)
{
  UpdateTM0(w0);
  UpdateTM1(w1);
  UpdateTMWithTM0(w1);
}

void MultilinearModel::UnfoldCoreTensor()
{
  tu0 = core.Unfold(0);
  tu1 = core.Unfold(1);
}
