# Clustering
A Simple C++ Clustering Library

### Features
* Sci-kit Learn like APIs
* Supoorted methods: KMenas, Spectral Clustering

### Usage
```C++
#include <iostream>
#include <string>

#include "../libclustering/Clustering.h"
#include "../libclustering/util/EigenIOHelper.h"

using namespace masc::clustering;
using namespace masc::clustering::util;

int main(int argc, char** argv) {
  MatrixXd X = EigenIOHelper::loadMatrix("noisy_moons.txt");
  KMeans kmeans;
  kmeans.setNClusters(2).setMaxIter(100).setNInit(20).setVerbosity(2);
  auto labels = kmeans.fit_predit(X);
  EigenIOHelper::save("labels.txt", labels);
  return 0;
}
```

### Build
* ./gen.sh && cd build/release/ && make

### Dependencies
* For use the lib:
  * Eigen >= 3
* For visualization:
  * matlibplot
  * numpy 
