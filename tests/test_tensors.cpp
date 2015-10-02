#define CATCH_CONFIG_MAIN
#include "../third_party/Catch/include/catch.hpp"

#include "../tensor.hpp"

TEST_CASE("Tensor construction", "[all tensors]") {
  CHECK_NOTHROW( Tensor1(10) );
  Tensor1 t(10);
  CHECK( t.size() == 10 );

  CHECK_NOTHROW( Tensor2(3, 2) );
  Tensor2 t2(3, 2);
  CHECK( t2.rows() == 3 );
  CHECK( t2.cols() == 2 );

  CHECK_NOTHROW(Tensor3(2, 3, 4));
  Tensor3 t3(2, 3, 4);
  CHECK( t3.layers() == 2 );
  CHECK( t3.rows() == 3 );
  CHECK( t3.cols() == 4 );
}

TEST_CASE("Tensor element accessors", "[all tensors]") {
  Tensor1 t(10);
  for(int i=0;i<t.size();++i) t(i) = i * i;
  for(int i=0;i<t.size();++i)
    CHECK( i*i == t(i) );

  Tensor2 t2(3, 2);
  for(int i=0, idx=0;i<t2.rows();++i) {
    for(int j=0;j<t2.cols();++j,++idx) {
      t2(i, j) = idx;
    }
  }
  const Tensor2 t2_ref{{0, 1}, {2, 3}, {4, 5}};
  CHECK( t2_ref == t2 );

  Tensor3 t3(2, 3, 4);
  for(int i=0, idx=0;i<t3.layers();++i) {
    for(int j=0;j<t3.rows();++j) {
      for(int k=0;k<t3.cols();++k, ++idx) {
        t3(i, j, k) = idx;
      }
    }
  }
  const Tensor3 t3_ref{{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                       {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}};
  CHECK( t3_ref == t3 );
}

TEST_CASE("Tensor unfolding/folding", "[all tensors]") {
  Tensor2 t2{{0, 1}, {2, 3}, {4, 5}};

  auto t2_unfolded = t2.Unfold();
  for(int i=0;i<t2_unfolded.size();++i) {
    CHECK( i == t2_unfolded(i));
  }

  Tensor3 t3{ { {0, 1, 2, 3},
                {4, 5, 6, 7},
                {8, 9, 10, 11} },
              { {12, 13, 14, 15},
                {16, 17, 18, 19},
                {20, 21, 22, 23} } };

  Tensor2 t3_unfold0 = t3.Unfold(0);
  CHECK( 2 == t3_unfold0.rows() );
  CHECK( 12 == t3_unfold0.cols() );
  const Tensor2 t3_unfold0_ref{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                               {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}};
  CHECK( t3_unfold0 == t3_unfold0_ref );

  Tensor3 t3new = Tensor3::Fold<0>(t3_unfold0, 2, 3, 4);
  CHECK( t3new == t3 );

  Tensor2 t3_unfold1 = t3.Unfold(1);
  CHECK( 3 == t3_unfold1.rows() );
  CHECK( 8 == t3_unfold1.cols() );
  const Tensor2 t3_unfold1_ref{{0, 12, 1, 13, 2, 14, 3, 15},
                               {4, 16, 5, 17, 6, 18, 7, 19},
                               {8, 20, 9, 21, 10, 22, 11, 23}};
  CHECK( t3_unfold1 == t3_unfold1_ref);
  Tensor3 t3new2 = Tensor3::Fold<1>(t3_unfold1, 2, 3, 4);
  CHECK( t3new2 == t3 );

  Tensor2 t3_unfold2 = t3.Unfold(2);
  CHECK( 4 == t3_unfold2.rows() );
  CHECK( 6 == t3_unfold2.cols() );
  const Tensor2 t3_unfold2_ref{{0, 4, 8, 12, 16, 20},
                               {1, 5, 9, 13, 17, 21},
                               {2, 6, 10, 14, 18, 22},
                               {3, 7, 11, 15, 19, 23}};
  CHECK( t3_unfold2 == t3_unfold2_ref );
  Tensor3 t3new3 = Tensor3::Fold<2>(t3_unfold2, 2, 3, 4);
  CHECK( t3new3 == t3 );
}

TEST_CASE("Tensor mode products", "[all tensors]") {
  Tensor2 t2{{0, 1}, {2, 3}, {4, 5}};

  Tensor3 t3{ { {0, 1, 2, 3},
                {4, 5, 6, 7},
                {8, 9, 10, 11} },
              { {12, 13, 14, 15},
                {16, 17, 18, 19},
                {20, 21, 22, 23} } };

  Tensor3 tm0 = t3.ModeProduct(t2, 0);
  const Tensor3 tm0_ref{{{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}},
                        {{36, 41, 46, 51}, {56, 61, 66, 71}, {76, 81, 86, 91}},
                        {{60, 69, 78, 87}, {96, 105, 114, 123}, {132, 141, 150, 159}}};
  CHECK( tm0_ref == tm0 );

  Tensor2 t22{{7, 6, 9}, {3, 1, 15}, {10, 12, 9}};
  Tensor3 tm1 = t3.ModeProduct<1>(t22);
  const Tensor3 tm1_ref{{{96, 118, 140, 162}, {124, 143, 162, 181}, {120, 151, 182, 213}},
                        {{360, 382, 404, 426}, {352, 371, 390, 409}, {492, 523, 554, 585}}};
  CHECK( tm1_ref == tm1 );

  Tensor2 t23{{7, 6, 9, 3}, {1, 15, 10, 12}, {9, 13, 10, 11}};
  Tensor3 tm2 = t3.ModeProduct<2>(t23);
  const Tensor3 tm2_ref{{{33, 71, 66}, {133, 223, 238}, {233, 375, 410}},
                        {{333, 527, 582}, {433, 679, 754}, {533, 831, 926}}};
  CHECK( tm2_ref == tm2 );
}

TEST_CASE("Tensor SVD", "[Tensor3]") {
  Tensor3 t3{ { {0, 1, 2, 3},
                {4, 5, 6, 7},
                {8, 9, 10, 11} },
              { {12, 13, 14, 15},
                {16, 17, 18, 19},
                {20, 21, 22, 23} } };

  auto comp = t3.svd();
  auto tcore = std::get<0>(comp);
  auto tu0 = std::get<1>(comp);
  auto tu1 = std::get<2>(comp);
  auto tu2 = std::get<3>(comp);

  const Tensor3 tcore_ref{ { {-65.3061, 0.11599, 1.21627e-14, -3.38305e-15},
                             {-0.0422502, -1.0848, -1.11917e-16,  1.62346e-15},
                             {1.78091e-15, 4.2886e-16, -1.04096e-15, -9.72653e-18} },
                           { {0.0261176, 2.34689, 4.83942e-16, -1.21872e-16},
                             {-7.16435, -1.04234, 9.36714e-16, -7.91326e-17},
                             {2.00494e-15, 4.51746e-16, 1.70148e-16, -4.62982e-16} } };
  CHECK( (tcore - tcore_ref).norm() < 1e-4 );

  const Tensor2 tu0_ref{{-0.32631, -0.945263},
                       {-0.945263, 0.32631}};
  CHECK( (tu0 - tu0_ref).norm() < 1e-4 );

  const Tensor2 tu1_ref{{-0.408404, 0.816419, 0.408248},
                       {-0.563323, 0.126491, -0.816497},
                       {-0.718243, -0.563436, 0.408248}};
  CHECK( (tu1 - tu1_ref).norm() < 1e-4 );

  const Tensor2 tu2_ref{{-0.45054, -0.704992, 0.542551, 0.0750901},
                       {-0.482653, -0.258933, -0.77937, 0.304273},
                       {-0.514766, 0.187126, -0.0689127, -0.833817},
                       {-0.546879, 0.633185, 0.305732, 0.454454}};
  CHECK( (tu2 - tu2_ref).norm() < 1e-4 );

  auto trecon = tcore.ModeProduct(tu0, 0)
                     .ModeProduct(tu1, 1)
                     .ModeProduct(tu2, 2);
  CHECK( (trecon - t3).norm() < 1e-10 );


  vector<int> modes{0, 1, 2};
  vector<int> dims{2, 2, 2};
  auto comp2 = t3.svd(modes, dims);

  tcore = std::get<0>(comp2);
  auto tus = std::get<1>(comp2);
  CHECK( (tcore - (Tensor3{{{-65.3061,  0.11599},
                             {-0.0422502, -1.0848}},
                           {{0.0261176, 2.34689},
                             {-7.16435,   -1.04234}}})).norm() < 1e-4);

  trecon = tcore;
  for(size_t i=0;i<modes.size();i++) {
    auto& tui = tus[i];
    trecon = trecon.ModeProduct(tui, modes[i]);
  }
  CHECK( (trecon - t3).norm() < 1e-10 );
}