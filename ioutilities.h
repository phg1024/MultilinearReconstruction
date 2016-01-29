#ifndef MULTILINEARRECONSTRUCTION_IOUTILITIES_H
#define MULTILINEARRECONSTRUCTION_IOUTILITIES_H

#include "common.h"
#include "constraints.h"
#include "utils.hpp"

#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "parameters.h"

#include <QImage>

vector<string> ReadFileByLine(const string &filename) {
  ifstream fin(filename);
  vector<string> lines;
  while (fin) {
    string line;
    std::getline(fin, line);
    if (!line.empty())
      lines.push_back(line);
  }
  return lines;
}

vector<int> LoadIndices(const string &filename) {
  cout << "Reading indices from " << filename << endl;
  vector<string> lines = ReadFileByLine(filename);
  vector<int> indices(lines.size());
  std::transform(lines.begin(), lines.end(), indices.begin(),
                 [](const string &s) {
                   return std::stoi(s);
                 });
  cout << indices.size() << " landmarks loaded." << endl;
  return indices;
}

vector<vector<int>> LoadContourIndices(const string& filename) {
  cout << "Reading contour indices from " << filename << endl;
  vector<string> lines = ReadFileByLine(filename);
  vector<vector<int>> contour_indices(lines.size());
  std::transform(lines.begin(), lines.end(), contour_indices.begin(),
                 [](const string &line) {
                   vector<string> parts;
                   boost::algorithm::split(parts, line,
                                           boost::algorithm::is_any_of(" "),
                                           boost::algorithm::token_compress_on);
                   auto parts_end = std::remove_if(parts.begin(), parts.end(),
                                                   [](const string &s) {
                                                     return s.empty();
                                                   });
                   vector<int> indices(std::distance(parts.begin(), parts_end));
                   std::transform(parts.begin(), parts_end, indices.begin(),
                                  [](const string &s) {
                                    return std::stoi(s);
                                  });
                   return indices;
                 });
  return contour_indices;
}

namespace std {
istream& operator>>(istream& is, Constraint2D& c) {
  is >> c.data.x >> c.data.y;
  return is;
}
}

vector<Constraint2D> LoadConstraints(const string& filename) {
  cout << "Reading constraints from " << filename << endl;
  ifstream fin(filename);
  if(!fin) {
    cerr << "Failed to open file " << filename << endl;
    exit(-1);
  }
  int num_constraints;
  fin >> num_constraints;

  istream_iterator<Constraint2D> iter(fin);
  istream_iterator<Constraint2D> iter_end;
  vector<Constraint2D> constraints;
  std::copy(iter, iter_end, back_inserter(constraints));

  std::for_each(constraints.begin(), constraints.end(), [](Constraint2D& c) {
    c.vidx = -1;
    c.weight = 1.0;
    // The coordinates are one-based. Fix them.
    c.data.x -= 1.0;
    c.data.y -= 1.0;
    DEBUG_EXPR(cout << c.data.x << ',' << c.data.y << endl;)
  });

  cout << num_constraints << " constraints expected. "
       << constraints.size() << " constraints loaded." << endl;
  assert(num_constraints == constraints.size());
  return constraints;
}

pair<QImage, vector<Constraint2D>> LoadImageAndPoints(
  const string &image_filename, const string &pts_filename) {
  QImage img(image_filename.c_str());

  auto constraints = LoadConstraints(pts_filename);

  // Compute a proper scale so the distance between pupils is approximately 200
  double puple_distance = glm::distance(
    0.5 * (constraints[28].data + constraints[30].data),
    0.5 * (constraints[32].data + constraints[34].data));
  const double reference_distance = 250.0;
  double scale_ratio = reference_distance / puple_distance;

  // Scale the image
  img = img.scaled(img.width() * scale_ratio, img.height() * scale_ratio,
                   Qt::KeepAspectRatio, Qt::SmoothTransformation);

  cout << "image size: " << img.width() << "x" << img.height() << endl;

  // Preprocess constraints
  for (auto &constraint : constraints) {
    constraint.data = constraint.data * scale_ratio;
    constraint.data.y = img.height() - 1 - constraint.data.y;
  }

  return make_pair(img, constraints);
};

vector<pair<string, string>> ParseSettingsFile(const string &filename) {
  vector<string> lines = ReadFileByLine(filename);

  vector<pair<string, string>> image_points_filenames(lines.size());
  std::transform(lines.begin(), lines.end(), image_points_filenames.begin(),
                 [](const string &line) {
                   vector<string> parts;
                   boost::algorithm::split(parts, line,
                                           boost::algorithm::is_any_of(" "),
                                           boost::algorithm::token_compress_on);
                   auto parts_end = std::remove_if(parts.begin(), parts.end(),
                                                   [](const string &s) {
                                                     return s.empty();
                                                   });
                   assert(std::distance(parts.begin(), parts_end) == 2);
                   return make_pair(parts.front(), parts.back());
                 });
  return image_points_filenames;
}

ReconstructionResult LoadReconstructionResult(const string &filename) {
  ReconstructionResult result;
  ifstream fin(filename);
  if(!fin) {
    cerr << "Failed to load reconstruction result from " << filename << endl;
    exit(-1);
  }
  fin >> result.params_cam >> result.params_model;
  fin.close();
  return result;
}

#endif //MULTILINEARRECONSTRUCTION_IOUTILITIES_H
