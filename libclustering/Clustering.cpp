/*
 * KMeans.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: zxi
 */

#include "Clustering.h"
#include <Eigen/Eigenvalues>

#include <iostream>
#include <algorithm>
#include <cfloat>
#include <ctime>

namespace masc {
namespace clustering {

#define TIMING(code, verbosity, output) \
  { \
    auto s = clock(); \
    (code); \
    auto e = clock(); \
    if (this->m_verbosity >= (verbosity) ) \
      std::cout<< (output) << " takes " << (e-s)*1.0 / CLOCKS_PER_SEC << " s" << std::endl; \
  } \

template<class T>
VectorXi ClusteringBase<T>::labelsInertia(const MatrixXd& X,
    const MatrixXd& centriods, VectorXd* distances, double* inertia) {

  const int n_samples = X.rows();
  const int n_k = centriods.rows();

  *distances = VectorXd(n_samples);
  *inertia = 0.0;

  VectorXi labels(n_samples);

  for (int i = 0; i < n_samples; ++i) {
    double min_dist = FLT_MAX;
    int label = -1;
    for (int j = 0; j < n_k; ++j) {
      double dist = (X.row(i) - centriods.row(j)).norm();
      if (dist < min_dist) {
        label = j;
        min_dist = dist;
      }
    }

    *inertia += min_dist;
    labels(i) = label;
    (*distances)(i) = min_dist;
  }

  return labels;
}

template<class T>
KMeansBase<T>::KMeansBase(int n_clusters, int n_init, int max_iter, double tol) :
    m_n_clusters(n_clusters), m_n_init(n_init), m_max_iter(max_iter), m_tol(tol) {
//TODO
}

template<class T>
KMeansBase<T>::~KMeansBase() {
  // nothing to do here
}

template<class T>
MatrixXd KMeansBase<T>::initCentroids(const MatrixXd& X) {
  const int n_samples = X.rows();
  const int n_features = X.cols();

  MatrixXd centroids(m_n_clusters, n_features);

  MatrixXd best_centroids;
  VectorXd distences;
  double best_score = FLT_MAX;

  std::uniform_int_distribution<int> dist(0, n_samples);

  for (int r = 0; r < m_n_init; ++r) {

    for (int i = 0; i < m_n_clusters; ++i) {
      int sample = dist(this->m_rd);
      centroids.row(i) = X.row(sample);
    }

    double score;
    this->labelsInertia(X, centroids, &distences, &score);

    if (score < best_score) {
      best_score = score;
      best_centroids = centroids;
    }

  }

  return best_centroids;
}

template<class T>
MatrixXd KMeansBase<T>::updateCenters(const MatrixXd& X,
    const VectorXi& labels) {
  const int n_samples = X.rows();
  const int n_features = X.cols();

  MatrixXd centroids = Eigen::MatrixXd::Zero(m_n_clusters, n_features);
  VectorXi count = Eigen::VectorXi::Zero(m_n_clusters);

  for (int i = 0; i < n_samples; ++i) {
    centroids.row(labels[i]) += X.row(i);
    count(labels[i]) += 1;
  }

  for (int i = 0; i < m_n_clusters; ++i) {
    if (count(i) > 0)
      centroids.row(i) /= count(i);
  }

  return centroids;
}

template<class T>
T& KMeansBase<T>::fit(const MatrixXd& X) {
  auto centers = this->initCentroids(X);

  if (this->m_verbosity >= 2)
    std::cout << "KMeansBase::fit - init centers = " << std::endl << centers << std::endl;

  VectorXd distances;

  double best_inertia = FLT_MAX / 2.0;
  VectorXi best_labels;
  MatrixXd best_centers;

  for (int i = 0; i < m_max_iter; ++i) {
    auto old_centers = centers;
    double inertia = 0.0;
    auto labels = this->labelsInertia(X, old_centers, &distances, &inertia);
    centers = this->updateCenters(X, labels);

    if (this->m_verbosity >= 1)
      std::cout << "KMeansBase::fit - iter " << i << " inertia = " << inertia << std::endl;

    if (inertia < best_inertia) {
      best_inertia = inertia;
      best_labels = labels;
      best_centers = centers;
    }

    auto shift = (old_centers - centers).norm();

    if (shift * shift < this->m_tol) {
      if (this->m_verbosity >= 1)
        std::cout << "KMeansBase::fit - Converged at iteration " << i << std::endl;
      break;
    }
  }

  this->m_cluster_centers = best_centers;
  this->m_labels = this->labelsInertia(X, best_centers, &distances,
      &best_inertia);
  this->m_inertia = best_inertia;

  return static_cast<T&>(*this);
}

///////////////////////////////////////////////////////////////////////////////
// KMeans
///////////////////////////////////////////////////////////////////////////////
KMeans::KMeans(int n_clusters, int n_init, int max_iter, double tol) :
    KMeansBase<KMeans>(n_clusters, n_init, max_iter, tol) {

}

KMeans::~KMeans() {
  //TODO
}

///////////////////////////////////////////////////////////////////////////////
// SpectralClustering
///////////////////////////////////////////////////////////////////////////////
SpectralClustering::SpectralClustering(int n_clusters, double gamma,
    AffinityType affinity_type) :
    KMeansBase<SpectralClustering>(n_clusters), m_affinity_type(affinity_type), m_gamma(
        gamma) {
//TODO
}

SpectralClustering::~SpectralClustering() {
  //TODO
}

MatrixXd SpectralClustering::constructAffinityMatrix(const MatrixXd& X,
    AffinityType affinity_type) {

  const int n_samples = X.rows();

  MatrixXd W(n_samples, n_samples);

  switch (affinity_type) {
  case AffinityType::RBF:

    // compute pairwise distance
    for (int i = 0; i < n_samples; ++i) {
      W(i, i) = 1.0;
      for (int j = i + 1; j < n_samples; ++j) {
        double dist = (X.row(i) - X.row(j)).norm();
        W(i, j) = W(j, i) = exp(-(this->m_gamma) * dist * dist);
      }
    }

    break;
  default:
    std::cerr << "Unsupported affinity type " << (int) affinity_type
        << std::endl;
    break;
  }

  return W;
}

SpectralClustering& SpectralClustering::fit(const MatrixXd& X) {
  const int n_samples = X.rows();
  const int n_features = X.cols();

  // Affinity matrix
  MatrixXd W;
  TIMING(W = this->constructAffinityMatrix(X, this->m_affinity_type), 1,
      "SpectralClustering::fit - construct affinity matrix");

  // Degree matrix
  MatrixXd D = W.rowwise().sum().asDiagonal();

  // Laplacian matrix
  MatrixXd L = D - W;

  Eigen::SelfAdjointEigenSolver<MatrixXd> es(n_samples);
  TIMING(es.compute(L), 1, "SpectralClustering::fit - compute eigen vectors");

  const MatrixXd& evs = es.eigenvectors();

  // embedded matrix (n_samples * k)
  MatrixXd embed(n_samples, m_n_clusters);

  for (int i = 0; i < m_n_clusters; ++i)
    embed.col(i) = evs.col(i);

  // run kmeans clustering on embed
  TIMING(KMeansBase<SpectralClustering>::fit(embed), 1,
      "SpectralClustering::fit - kmeans clustering");

  return *this;
}

} /* namespace clustering */
} /* namespace masc */
