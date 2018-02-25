
## Project Page
[Multilinear Reconstruction](http://phg1024.github.io/MultilinearReconstruction)

## Data
Some relevant data could be downloaded [here](https://goo.gl/sEB9Dk).

## Dependencies
* Boost 1.63
* ceres-solver 1.12.0
* OpenCV 3.4
* freeglut
* GLEW
* Eigen 3.3.0
* Intel MKL
* SuiteSparse 4.5.3
* Qt5
* OpenMesh 6.3
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
