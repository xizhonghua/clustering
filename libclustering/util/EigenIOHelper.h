/*
 * EigenIOHelper.h
 *
 *  Created on: Mar 12, 2016
 *      Author: zxi
 */

#ifndef LIBCLUSTERING_UTIL_EIGENIOHELPER_H_
#define LIBCLUSTERING_UTIL_EIGENIOHELPER_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

namespace masc {
namespace clustering {
namespace util {

class EigenIOHelper {
public:

  template<class T>
  static void save(const std::string& filename, const T& m) {
    std::ofstream fout(filename);
    if (!fout.good()) {
      std::cerr << "!Error! Unable to open file:" << filename << std::endl;
    }

    fout << m << std::endl;
    fout.close();
  }

  static Eigen::MatrixXd loadMatrix(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin.good()) {
      std::cerr << "!Error! Unable to open file:" << filename << std::endl;
    }

    std::vector<std::vector<double>> vecs;

    std::string line;

    size_t max_cols = 0;

    while (std::getline(fin, line)) {
      std::stringstream ss(line);
      std::vector<double> vals;
      double val;
      while (ss >> val) {
        vals.push_back(val);
      }

      vecs.push_back(vals);

      max_cols = std::max(max_cols, vals.size());
    }

    Eigen::MatrixXd m(vecs.size(), max_cols);

    for (int i = 0; i < vecs.size(); ++i)
      for (int j = 0; j < vecs[i].size(); ++j)
        m(i, j) = vecs[i][j];

    //fin >> m;
    fin.close();

    return m;
  }
};

}
} /* namespace clustering */
} /* namespace masc */

#endif /* LIBCLUSTERING_UTIL_EIGENIOHELPER_H_ */
