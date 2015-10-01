#include "../tensor.hpp"

inline void testTensors() {
  Tensor1 t(10);
  for(int i=0;i<t.size();i++) {
    t(i) = (double)rand();
  }

  cout << t << endl;

  Tensor2 t2(3, 2);
  for(int i=0;i<t2.rows();i++) {
    for(int j=0;j<t2.cols();j++) {
      t2(i, j) = (double)(rand() % 16);
    }
  }

  t2.print("T2");

  cout << "T2 unfolded:\n" << t2.Unfold() << endl;


  Tensor3 t3(2, 3, 4);
  for(int i=0;i<t3.layers();i++) {
    for(int j=0;j<t3.rows();j++) {
      for(int k=0;k<t3.cols();k++) {
        t3(i, j, k) = (double)(rand() % 32 );
      }
    }
  }

  t3.print("T3");

  cout << "unfold in mode 0:" << endl;
  Tensor2 t3_unfold0 = t3.Unfold(0);
  t3_unfold0.print("T30");
  Tensor3 t3new = Tensor3::Fold<0>(t3_unfold0, 2, 3, 4);
  t3new.print("fold back");

  cout << "unfold in mode 1:" << endl;
  Tensor2 t3_unfold1 = t3.Unfold(1);
  t3_unfold1.print("T31");
  Tensor3 t3new2 = Tensor3::Fold<1>(t3_unfold1, 2, 3, 4);
  t3new2.print("fold back");

  cout << "unfold in mode 1:" << endl;
  Tensor2 t3_unfold2 = t3.Unfold(2);
  t3_unfold2.print("T32");
  Tensor3 t3new3 = Tensor3::Fold<2>(t3_unfold2, 2, 3, 4);
  t3new3.print("fold back");

  cout << "mode product" << endl;
  Tensor3 tm0 = t3.ModeProduct(t2, 0);
  tm0.print("TM0");

  Tensor2 t22(3, 3);
  for(int i=0;i<t22.rows();i++) {
    for(int j=0;j<t22.cols();j++) {
      t22(i, j) = (double)(rand() % 16);
    }
  }
  t22.print("T22");

  Tensor3 tm1 = t3.ModeProduct<1>(t22);
  tm1.print("TM1");

  Tensor2 t23(3, 4);
  for(int i=0;i<t23.rows();i++) {
    for(int j=0;j<t23.cols();j++) {
      t23(i, j) = (double)(rand() % 16);
    }
  }
  t23.print("T23");

  Tensor3 tm2 = t3.ModeProduct<2>(t23);
  tm2.print("TM2");

  auto comp = t3.svd();
  auto tcore = std::get<0>(comp);
  auto tu0 = std::get<1>(comp);
  auto tu1 = std::get<2>(comp);
  auto tu2 = std::get<3>(comp);

  tcore.print("core");
  tu0.print("u0");
  tu1.print("u1");
  tu2.print("u2");

  auto trecon = tcore.ModeProduct(tu0, 0)
    .ModeProduct(tu1, 1)
    .ModeProduct(tu2, 2);
  trecon.print("recon");
  t3.print("ref");

  int ms[3] = {0, 1, 2};
  int ds[3] = {2, 3, 4};
  vector<int> modes(ms, ms+3);
  vector<int> dims(ds, ds+3);
  auto comp2 = t3.svd(modes, dims);

  tcore = std::get<0>(comp2);
  auto tus = std::get<1>(comp2);
  tcore.print("core");
  trecon = tcore;
  for(size_t i=0;i<modes.size();i++) {
    auto& tui = tus[i];
    trecon = trecon.ModeProduct(tui, modes[i]);
    tui.print("ui");
  }

  trecon.print("recon");
  t3.print("ref");
}
