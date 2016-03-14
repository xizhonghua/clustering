/*
 * kmeans.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: zxi
 */

#include <iostream>
#include <string>
#include <fstream>

#include <cstdlib>
#include <ctime>

#include "../libclustering/Clustering.h"
#include "../libclustering/util/EigenIOHelper.h"

using namespace masc::clustering;
using namespace masc::clustering::util;

int main(int argc, char** argv) {

  srand(time(nullptr));

  // load dataset
  MatrixXd X = EigenIOHelper::loadMatrix("noisy_moons.txt");

  std::cout << "dataset size " << X.rows() << "*" << X.cols() << std::endl;

  const int n_samples = X.rows();
  const int n_features = X.cols();

  auto start = clock();
  KMeans kmeans;
  kmeans.setNClusters(2).setMaxIter(100).setNInit(20).setTol(0.00001).setVerbosity(
      2);
  auto kmeans_labels = kmeans.fit_predit(X);
  std::cout << "kmeans takes " << (clock() - start) * 1.0 / CLOCKS_PER_SEC
      << " s" << std::endl;

  EigenIOHelper::save("labels_kmeans.txt", kmeans_labels);

  start = clock();
  SpectralClustering sc(2);
  sc.setGamma(30).setVerbosity(1);
  auto sc_labels = sc.fit_predit(X);
  std::cout << "sc takes " << (clock() - start) * 1.0 / CLOCKS_PER_SEC << " s"
      << std::endl;

  EigenIOHelper::save("labels_sc.txt", sc_labels);
}

