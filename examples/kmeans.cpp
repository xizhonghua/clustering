/*
 * kmeans.cpp
 *
 *  Created on: Mar 13, 2016
 *      Author: zxi
 */
#include <iostream>
#include "libclustering/Clustering.h"
using namespace masc::clustering;

int main(int argc, char** argv) {
  auto X = Eigen::MatrixXd::Random(100, 3);
  KMeans kmeans(4);
  auto labels = kmeans.fit_predit(X);

  for (int i = 0; i < labels.rows(); ++i)
    std::cout << X.row(i) << " " << labels(i) << std::endl;
}

