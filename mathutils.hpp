#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

template <typename T>
T deg2rad(T val) {
  return val / 180.0 * 3.1415926;
}

template <typename T>
T rad2deg(T val) {
  return val / 3.1415926 * 180.0;
}

#endif // MATHUTILS_HPP
