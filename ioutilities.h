#ifndef MULTILINEARRECONSTRUCTION_IOUTILITIES_H
#define MULTILINEARRECONSTRUCTION_IOUTILITIES_H

#include "common.h"
#include "constraints.h"
#include "utils.hpp"

#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "parameters.h"

#include <QImage>

vector<string> ReadFileByLine(const string &filename);
vector<int> LoadIndices(const string &filename);

vector<vector<int>> LoadContourIndices(const string& filename);

namespace std {
inline istream& operator>>(istream& is, Constraint2D& c);
}

vector<Constraint2D> LoadConstraints(const string& filename);
pair<QImage, vector<Constraint2D>> LoadImageAndPoints(
  const string &image_filename, const string &pts_filename, bool resize=true);
vector<pair<string, string>> ParseSettingsFile(const string &filename);
ReconstructionResult LoadReconstructionResult(const string &filename);

#endif //MULTILINEARRECONSTRUCTION_IOUTILITIES_H
