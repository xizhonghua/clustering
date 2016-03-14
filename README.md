# Clustering
A Simple C++ Clustering Library

### Features
* Sci-kit Learn like APIs
* Supoorted methods: KMenas, Spectral Clustering

### Usage
```C++
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
```

### Build
```bash
./gen.sh && cd build/release/ && make
```

### Dependencies
* For use the lib:
  * Eigen >= 3
* For visualization:
  * matplotlib
  * numpy 
