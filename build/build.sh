if [ "$2" = "intel" ]; then
  echo "Using intel compilers"
  cmake .. -DCMAKE_BUILD_TYPE=$1 -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc
else
  echo "Using GCC compilers"
  cmake .. -DCMAKE_BUILD_TYPE=$1 -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
fi

make -j4
