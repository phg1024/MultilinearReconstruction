#include "multilinearmodel.h"

MultilinearModel::MultilinearModel(const string &filename)
{
  core.read(filename);
  unfold();
}

MultilinearModel MultilinearModel::project(const vector<int> &indices)
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

  newmodel.unfold();

  return newmodel;
}

void MultilinearModel::updateTM0(const Tensor1 &w)
{
  tm0 = core.modeProduct<0>(w);
}

void MultilinearModel::updateTM1(const Tensor1 &w)
{
  tm1 = core.modeProduct<1>(w);
}

void MultilinearModel::updateTMWithTM0(const Tensor1 &w)
{
  tm = tm0.modeProduct<0>(w);
}

void MultilinearModel::updateTMWithTM1(const Tensor1 &w)
{
  tm = tm1.modeProduct<0>(w);
}

void MultilinearModel::applyWeights(const Tensor1 &w0, const Tensor1 &w1)
{
  updateTM0(w0);
  updateTM1(w1);
  updateTMWithTM0(w1);
}

void MultilinearModel::unfold()
{
  tu0 = core.unfold(0);
  tu1 = core.unfold(1);
}
