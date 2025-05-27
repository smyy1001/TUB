"""
Solution to task 2 of Exercise 4: Clustering with DBSCAN 

(C) Merten Stender, TU Berlin, 2025
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as DBSCAN
from sklearn.metrics import silhouette_score as sil_coeff
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def load_dataset() -> np.ndarray:
    # Load example data for clustering (code given in assignment)
    # assumes the files in the same directory as this script

    data_path = os.path.join(os.path.dirname(__file__),
                             'secondary_hand_car_sales.npy')
    data = np.load(data_path, allow_pickle=True)

    # extract year, mileage, price
    data = data[:, -3:]
    return data.astype(np.float32)


def num_clusters(labels: np.ndarray) -> int:
    """
    Count the number of clusters in the labels array.
    """
    return len(set(labels)) - (1 if -1 in labels else 0)


if __name__ == "__main__":
    # Load example data for clustering (code given in assignment)
    data = load_dataset()

    # Print the shape of the data, and the mean and standard deviation
    # per column
    print(f"\n\nData shape: {data.shape}")
    print(f"Mean of data: {np.mean(data, axis=0)}")
    print(f"Std. of data: {np.std(data, axis=0)}")

    # zscore the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    print(f"Mean of z-scaled data: {np.mean(data_scaled, axis=0)}")

    # cluster the dataset using DBSCAN (default parameters)
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(data_scaled)

    print(f'number of clusters: {num_clusters(labels)}')
    print(f'number of outliers: {np.sum(labels==-1)}')

    # # hyperparameter search for N_min, epsilon
    # nmin_grid = np.arange(10, 11)
    # epsilon_grid = np.linspace(0.25, 0.8, num=25)

    # silhouette_values = []
    # number_of_clusters = []
    # for _nmin in nmin_grid:
    #     for _epsilon in epsilon_grid:

    #         _dbscan = DBSCAN(eps=_epsilon, min_samples=_nmin)
    #         _labels = _dbscan.fit_predict(data_scaled)

    #         _num_clusters = num_clusters(_labels)
    #         if _num_clusters > 1:
    #             _silhouette_value = sil_coeff(data_scaled, _labels)
    #         else:
    #             _silhouette_value = -1

    #         silhouette_values.append(_silhouette_value)
    #         number_of_clusters.append(_num_clusters)

    # silhouette_values = np.array(silhouette_values)
    # number_of_clusters = np.array(number_of_clusters)

    # # extract best clustering w.r.t. silh. score
    # idx_best = np.argmax(silhouette_values)
    # best_epsilon = epsilon_grid[idx_best]
    # best_silhouette_val = silhouette_values[idx_best]

    # print(f'best clustering: num. clusters: {number_of_clusters[idx_best]} eps={best_epsilon}, sil. value= {best_silhouette_val}')

    # # visualize the hyperparameter results
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(epsilon_grid, silhouette_values)
    # plt.xlabel(r'$\epsilon$')
    # plt.ylabel('silhouette value')

    # plt.subplot(1,2,2)
    # plt.plot(epsilon_grid, number_of_clusters)
    # plt.xlabel(r'$\epsilon$')
    # plt.ylabel('number of clusters')
    # plt.savefig('hyperparameter_search.png')
    # plt.show()

    # visualize the best clustering, i.e. using the optimal 
    # hyperparameters
    best_epsilon = 0.25
    dbscan = DBSCAN(eps=best_epsilon, min_samples=10)
    labels = dbscan.fit_predict(data_scaled)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel('year of manufacture')
    plt.ylabel('mileage')
    plt.savefig('best_clustering.png')
    plt.show()




