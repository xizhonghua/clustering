/*
 * ClusterBase.h
 *
 *  Created on: Mar 12, 2016
 *      Author: zxi
 */

#ifndef LIBCLUSTERING_CLUSTERING_H_
#define LIBCLUSTERING_CLUSTERING_H_

#include <random>
#include <Eigen/Dense>

namespace masc {
namespace clustering {
using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;

enum class AffinityType {
  RBF, PRE_COMPUTE
};

template<class T>
class ClusteringBase {
public:
  ClusteringBase() :
      m_inertia(0.0), m_verbosity(0) {

  }

  virtual ~ClusteringBase() {
  }

  virtual T& fit(const MatrixXd& X) = 0;

  const VectorXi& fit_predit(const MatrixXd& X) {
    this->fit(X);
    return this->getLabels();
  }

  /////////////////////////////////////////////////////////////////////////////
  // setter/getter
  /////////////////////////////////////////////////////////////////////////////
  const MatrixXd& getClusterCenters() const {
    return m_cluster_centers;
  }
  virtual const VectorXi& getLabels() const {
    return m_labels;
  }

  const double getInertia() const {
    return m_inertia;
  }

  T& setVerbosity(int verbosity) {
    this->m_verbosity = verbosity;
    return static_cast<T&>(*this);
  }

protected:

  // assign the labels for each sample based on the centriods using X^2 distance
  // output parameters:
  //    distances
  //    inertia
  // return:
  //    labels (n_samples * 1)
  VectorXi labelsInertia(const MatrixXd& X, const MatrixXd& centriods,
      VectorXd* distances, double* inertia);

  std::random_device m_rd;

  int m_verbosity;              // 0 = quiet, 1 = verbose ....
  MatrixXd m_cluster_centers;   // n_clusters * n_feature
  VectorXi m_labels;            // n_samples * 1
  double m_inertia;

};

template<class T>
class KMeansBase: public ClusteringBase<T> {
public:
  /////////////////////////////////////////////////////////////////////////////
  // constructor and decosntructor
  /////////////////////////////////////////////////////////////////////////////
  KMeansBase(int n_clusters = 8, int n_init = 10, int max_iter = 300,
      double tol = 0.0001);
  virtual ~KMeansBase();

  /////////////////////////////////////////////////////////////////////////////
  // setter/getter
  /////////////////////////////////////////////////////////////////////////////
  virtual T& fit(const MatrixXd& X) override;

  // Set number of clusters
  T& setNClusters(int n_clusters) {
    this->m_n_clusters = n_clusters;
    return static_cast<T&>(*this);
  }

  // Set number of center initialization trials
  T& setNInit(int n_init) {
    this->m_n_init = n_init;
    return static_cast<T&>(*this);
  }

  // Set maximum iterations
  T& setMaxIter(int max_iter) {
    this->m_max_iter = max_iter;
    return static_cast<T&>(*this);
  }

  // Set centroids shift threshold
  T& setTol(double tol) {
    this->m_tol = tol;
    return static_cast<T&>(*this);
  }

protected:

  MatrixXd initCentroids(const MatrixXd& X);
  MatrixXd updateCenters(const MatrixXd& X, const VectorXi& labels);

  int m_n_clusters;
  int m_n_init;
  int m_max_iter;
  double m_tol;

};

class KMeans: public KMeansBase<KMeans> {
public:
  KMeans(int n_clusters = 8, int n_init = 10, int max_iter = 300, double tol =
      0.0001);
  virtual ~KMeans();
};

class SpectralClustering: public KMeansBase<SpectralClustering> {
public:
  /////////////////////////////////////////////////////////////////////////////
  // constructor and decosntructor
  /////////////////////////////////////////////////////////////////////////////
  SpectralClustering(int n_clusters = 4, double gamma = 1.0,
      AffinityType affinity_type = AffinityType::RBF);
  virtual ~SpectralClustering();

  /////////////////////////////////////////////////////////////////////////////
  // Methods
  /////////////////////////////////////////////////////////////////////////////

  /**
   * compute the spectral clustering of dataset
   * input:
   *   X: pairwise distance matrix, (n_samples * n_samples)
   */
  virtual SpectralClustering& fit(const MatrixXd& X) override;

  /////////////////////////////////////////////////////////////////////////////
  // setter/getter
  /////////////////////////////////////////////////////////////////////////////
  SpectralClustering& setGamma(double gamma) {
    this->m_gamma = gamma;
    return *this;
  }
protected:

  MatrixXd constructAffinityMatrix(const MatrixXd& X,
      AffinityType affinity_type);

  // affinity type
  AffinityType m_affinity_type;

  // gamma for RBF affinity
  // affinity(i,j) = exp(-gamma * dist(i,j)^2)
  double m_gamma;
};

} /* namespace clustering */
} /* namespace masc */

#endif /* LIBCLUSTERING_CLUSTERING_H_ */
