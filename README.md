
## Project Page
[Multilinear Reconstruction](http://phg1024.github.io/MultilinearReconstruction)

## Data
Some relevant data could be downloaded [here](https://goo.gl/sEB9Dk).

## Dependencies
* Boost
* ceres-solver
* OpenCV
* freeglut
* GLEW
* Eigen
* Intel MKL
* SuiteSparse
* Qt5
* OpenMesh
* PhGLib: https://github.com/phg1024/PhGLib.git

## Compile
```bash
git clone --recursive https://github.com/phg1024/MultilinearReconstruction.git
cd MultilinearReconstruction
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release ..
make -j8
```
