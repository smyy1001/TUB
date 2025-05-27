"""
Solution to exercise 03: k-means

DISCLAIMER:
Please note that this solution may contain errors (please report them, thanks!), and that there are most-likely more elegant and more efficient implementations available for the given problem. In this light, this solution may only serve as an inspiration for solving the given exercise tasks.

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""


import numpy as np
from utils_clustering import pick_random_points, sse, bss, find_farthest_point
from utils_clustering import plot_clusters
from matplotlib import pyplot as plt


def assign_cluster(x: np.ndarray, centroids: np.ndarray, norm: str = 'L2') -> np.ndarray:
    """Assign a cluster index vector to data points given some centroids.

    x:          [N, n]  all data points in n-dimensional space
    centroids:  [K, n]  K centroid coordinates
    norm:       string for the norm: 'l2' or 'l1'

    returns     [N,1] cluster label assignments {0, .. (K-1)}

    The order of the centroid will give the numbering of the labels, starting from 0
    """

    # number of clusters
    K = centroids.shape[0]

    # select the distance norm to be used
    if norm == 'L1':
        norm_ord = 1
    elif norm == 'L2':
        norm_ord = 2
    else:
        raise ValueError('norm must be L1 or L2!')

    dists = []  # storing distances from all points to all centroids
    for k in range(K):  # iterate over all K centroids

        # compute distance from all x to current centroid m_k
        dists.append(np.linalg.norm(x-centroids[k, :], ord=norm_ord, axis=1))

    dists = np.vstack(dists)  # shape: [K, N]

    # find the row index of minimum per column, i.e. the index of the closest centroid
    cluster_labels = np.argmin(dists, axis=0)  # shape: N, 1

    return cluster_labels


def update_centroids(x: np.ndarray, labels: np.ndarray, K: int, norm: str = 'L2') -> np.ndarray:
    """Compute new centroid coordinates based on averaging across cluster members.

    x:      data points [N, n]  all data points in n-dimensional space
    labels: cluster assignment vector [N,1] contains {0, ... K-1}; zero-indexed!
    K:      number of clusters K. required as there can also be empty clusters if centroid far away
    norm:   string for the norm: 'L2' or 'L1': mean or median

    returns
    centroids: [K,n] new centroid coordinates

    """
    centroids_new = []

    # loop over clusters and re-compute cluster coordinates by averaging
    for k in range(K):

        # boolean index, true when data point belongs to current cluster k
        in_cluster = labels == k

        # check if cluster is empty. if so, re-locate the centroid in order to
        # keep the clustering going
        if any(in_cluster):

            # compute mean coordinates across all d dimensions for data points in cluster
            if norm == 'L2':
                # mean results for sum of squares
                centroids_new.append(np.mean(x[in_cluster], axis=0))
            elif norm == 'L1':
                # medin results for sum of differences
                centroids_new.append(np.median(x[in_cluster], axis=0))

        elif any(in_cluster) is False:  # no data point assigned to this cluster
            print(f'Iteration {k}: ATTENTION! At least one cluster is empty')
            if not centroids_new:  # check if we have existing centroids
                existing_centroids = None
            else:
                existing_centroids = np.vstack(centroids_new)

            centroids_new.append(relocate_empty_centroid(
                x=x, centroids=existing_centroids))
            print(f'relocated empty centroid to {centroids_new[-1]}')

    centroids = np.vstack(centroids_new)

    return centroids


def is_converged(centroids: list, labels: list) -> bool:
    """Determine if K-means is converged.

    centroids: [n_iters, [np.ndarray[K, n]]] for the number of previous iterations n_iters, K clusters in n dimensions
    r: [n_iters, [np.ndarray[N,]]] assignment vectors for previous iterations and N data points. contains values 1, .., K

    returns boolean. True if convergence criterion is matched.
    """

    n_iters = len(centroids)    # number of K-means iterations
    N = labels[0].shape         # number of data points

    # we require the algorithm to run at least for 5 iterations
    if n_iters < 5:
        converged = False
    else:

        # we compute the number of points that change the cluster assignment from previous to current iteration
        prev_r = labels[0]  # previous assignment vector
        state = []          # converged?

        for it in range(n_iters):
            curr_r = labels[it]  # current assignment vector

            # compute number of elements that changed
            ratio_changed = (N - np.sum(prev_r == curr_r)) / N
            state.append(ratio_changed < 0.01)
            prev_r = curr_r

        # check if all last <5> states are true, i.e. fullfill the condition. If yes: return True
        converged = all(state[-5:])

    return converged


def relocate_empty_centroid(x: np.ndarray, centroids: np.ndarray = None) -> np.ndarray:
    """ Relocate an empty centroid

    We are placing the centroid as far as possible away from any existing centroid

    x: [N, n] N data points in n-dimensional data space
    centroids: [M, n] M existing centroids

    returns
    centroid_new: [1, n] coordinate of a new centroid placed farthes away from
    existing centroids
    """

    if centroids is not None:
        centroid_new = find_farthest_point(x=x, y=centroids)
    else:  # select the sample mean if no other centroids exist
        centroid_new = np.mean(x, axis=0)

    return centroid_new


def kmeans_clustering(x: np.ndarray, K: int, norm: str = 'L2', init_centroids: np.ndarray = None):
    """Basic K-means algorithm.

    x:      [N, n] N data points in n-dimensional data space
    K:      int, number of clusteres desired to find
    norm:   str, ['L2', 'L1'] distance metric to consider

    returns
    labels      [N,1]  final labels
    centroids   [K, n]  finale centroids
    cost        [n_iters]  cost value for each iteration
    """

    # the distance norm to use
    if (norm != 'l1') and (norm != 'L2'):
        raise ValueError('invalid norm, use L1 or L2!')

    # initialization of centroids
    if init_centroids is None:

        # randomly selecting K points from the data points
        centroids_0 = pick_random_points(x=x, K=K)
    else:
        centroids_0 = init_centroids

    # initialize the return values (list along the iteration)
    centroids = [centroids_0]
    labels = [np.zeros(x.shape[0])]
    cost_vals = {'sse': [], 'bss': []}

    converged = False
    i = 0

    # Both subexpressions must be true for the compound expression to be considered
    # true. If one subexpression is false, then the compound expression is false
    while (not converged) and (i < 100):  # make sure to catch some infinite looping!

        # print(f'K-means iteration {i}')

        # Phase 1: update cluster assignment
        labels.append(assign_cluster(x=x, centroids=centroids[-1], norm=norm))

        # Phase 2: update centroid positions
        centroids.append(update_centroids(
            x=x, labels=labels[-1], K=K, norm=norm))

        # Check convergence criterion
        # do not consider the initialization
        converged = is_converged(centroids=centroids[1:], labels=labels[1:])
        # print(f'convergence criterion is: {converged}')

        # supplementary logging of cost value
        cost_vals['sse'].append(
            sse(x=x, centroids=centroids[-1], labels=labels[-1], norm=norm))
        cost_vals['bss'].append(bss(x=x, centroids=centroids[-1], norm=norm))
        # print(f'cost values are: SSE={cost_vals["sse"][-1]}, \t BSS={cost_vals["bss"][-1]}\n')

        i += 1

    return labels[-1], centroids[-1], cost_vals


if __name__ == "__main__":

    """ let's test the functionalities """

    # data points
    x = np.array([[1, 9], [2, 7], [3, 8],
                  [4, 3], [5, 2], [7, 2], [8, 4], [6, 4],
                  [10, 11], [10, 9], [8, 11], [12, 9]])

    # plot the raw data without any clustering
    plot_clusters(x=x)

    # ground truth labels
    labels_gt = np.array([0, 0, 0,
                          1, 1, 1, 1, 1,
                          2, 2, 2, 2])

    centroids_gt = update_centroids(x=x, labels=labels_gt, K=3, norm='L2')
    
    # ground truth centroids
    centroids_gt = np.array([[2, 8], [6, 3], [10, 10]])

    # initial centroids
    centroids_0 = np.array([[4, 4], [5, 5], [6, 6]])

    # TASK 1: update cluster assigment and plot result
    labels_0 = assign_cluster(x=x, centroids=centroids_0, norm='L2')
    plot_clusters(x=x, labels=labels_0, centroids=centroids_0)

    # Task 2: update centroid positions and plot results
    centroids_1 = update_centroids(x=x, labels=labels_0, K=3, norm='L2')
    plot_clusters(x=x, labels=labels_0, centroids=centroids_1)

    print(f'old centroid positions: \n{centroids_0} \n')
    print(f'new centroid positions: \n{centroids_1}')

    # call the K-means clustering algorithm
    labels, centroids, cost_val = kmeans_clustering(
        x=x, K=3, norm='L2', init_centroids=centroids_0)

    # plot the result
    plot_clusters(x=x, labels=labels, centroids=centroids)

    """
    Extra tasks
    """

    # load csv file and plot data
    data = np.genfromtxt('example_data_Kmeans.csv', delimiter=',')
    plot_clusters(x=data)

    from sklearn.cluster import KMeans as KMeans

    # obtain the clustering 
    kmeans = KMeans(n_clusters=4, n_init="auto").fit(data)
    centroids_sklearn = kmeans.cluster_centers_
    labels_sklearn = kmeans.labels_
    
    # plot the results
    plot_clusters(x=data, centroids=centroids_sklearn, labels=labels_sklearn)
    plt.show()
    print(f'\n\nInertia of the sklearn clustering: \t\t\t\t inertia = {kmeans.inertia_:.2f}')
    
    
    # using my own implementation
    labels_self, centroids_self, _ = kmeans_clustering(x=data, K=4, norm='L2')
    plot_clusters(x=data, centroids=centroids_self, labels=labels_self)
    plt.show()
    SSE_self = sse(x=data, centroids=centroids_self, labels=labels_self)
    print(f'Inertia of the self implemented clustering: \t inertia = {SSE_self:.2f}')
    
    
