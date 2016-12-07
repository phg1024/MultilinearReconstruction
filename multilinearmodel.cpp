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
#if 0
  tm0 = core.ModeProduct<0>(w);
#else
  // tu0
  // id0: | exp0 | exp1 | ... | expn |
  // id1: | exp0 | exp1 | ... | expn |
  // ...
  // idn: | exp0 | exp1 | ... | expn |

  auto tm0u = tu0.ModeProduct<0>(w);
  tm0 = Tensor2::FoldByColumn(tm0u, core.rows(), core.cols());
#endif
}

void MultilinearModel::UpdateTM1(const Tensor1 &w)
{
#if 0
  tm1 = core.ModeProduct<1>(w);
#else
  // tu1
  // exp0: | x0 | y0 | z0 | ..
  // exp1:
  // ...
  // expn:
  auto tm1u = tu1.ModeProduct<0>(w);
  tm1 = Tensor2::FoldByRow(tm1u, core.layers(), core.cols());
#endif
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
