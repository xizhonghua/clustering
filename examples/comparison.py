#!/usr/bin/python
import matplotlib.pyplot as plot
import numpy as np


def plot_clusters(data_filename, labels_filename, method):
  data = np.loadtxt(data_filename)
  labels = np.loadtxt(labels_filename)
  for i in range(int(labels.min()), int(labels.max()) + 1):
    plot.scatter(
        data[
            labels == i, 0], data[
            labels == i, 1], color=np.random.random(
            (1, 3)))
  plot.title(method)
  plot.show()

plot_clusters('noisy_moons.txt', 'labels_kmeans.txt', 'kmenas')
plot_clusters('noisy_moons.txt', 'labels_sc.txt', 'speactral clustering')
