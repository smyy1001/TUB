"""
Solution to exercise 05: implementing helper functions for building a 
decision tree from scratch

DISCLAIMER:
Please note that this solution may contain errors (please report them, thanks!), and that there are most-likely more
elegant and more efficient implementations available for the given problem. In this light, this solution may only
serve as an inspiration for solving the given exercise tasks.

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""
import matplotlib.pyplot as plt
import numpy as np


def entropy(y: np.ndarray) -> float:
    """
    Computes Shannon entropy for a (label) vector y
    """
    # Pythonic way to count occurences
    # proportions = np.bincount(y) / len(y)  # works only for integer labels!
    proportions = np.unique(y, return_counts=True)[1] / len(y)  # works for any labels
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])


def information_gain(y_parent: np.ndarray, index_split: np.ndarray) -> float:
    """
    Computes information gain (parent's entropy - weighted children's entropy)
    for a given split provided in
    index_split: binary vector (1 for left child, 0 for right child)
    y_parent: label vector in the parent node
    """

    # compute number of members per child node according to split index
    n = len(index_split)  # overall number of data points
    n_left = np.sum(index_split == 1)  # members of left child
    n_right = np.sum(index_split == 0)  # members of right child

    # information gain will be zero if a child has no members (special case)
    if n_left == 0 or n_right == 0:
        return 0

    else:

        # compute entropy at parent node (which the split should reduce)
        entropy_parent = entropy(y_parent)

        # child entropies and weight according to relative node size
        entropy_left = entropy(y_parent[index_split])  
        entropy_right = entropy(y_parent[index_split == 0])

        weighted_child_entropy = (n_left / n) * entropy_left + \
                                 (n_right / n) * entropy_right

        # return information gain
        return entropy_parent - weighted_child_entropy


def create_split(x: np.ndarray, split_dim: np.ndarray, split_val: np.ndarray) -> np.ndarray:
    """
    Splits a given data set along a split dimension 
    and a split value. 

    Expects the data set x of shape [N, n], i.e. number of samples x num features
    Returns a binary index vector where x[split_dim] <= split_value is true
    """

    # we need to check that the splitting dimension is valid
    if split_dim >= x.shape[1]:
        raise ValueError('splitting dimension exceeds number of features')

    # create binary vector, true entry corresponds to left child,
    # false to right child
    return x[:, split_dim] <= split_val


def best_split(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Find the best binary feature space segmentation according to 
    Shannons entropy and the information gain. 
    
    Expects the data set x of shape [N, n], and the
    label vector of shape [N], carrying integer/boolean class labels
    """

    # we'll iterate through all possible splits along all feature
    # space dimensions. To keep track of the results, we simply compare
    # the current split and resulting info.gain against the best so far.
    best_info_gain = 0
    best_dim = None
    best_thresh = None

    # loop through all feature space dimensions
    for _dim in range(x.shape[1]):

        # find all possible splitting values in current feature dimension
        thresholds = np.unique(x[:, _dim])

        # loop through possible splits in current feat. dimension
        for _thresh in thresholds:

            # create split and compute information gain
            _index_split = create_split(x, split_dim=_dim, split_val=_thresh)
            _score = information_gain(y, _index_split)
            
            # update if score was better than before
            if _score > best_info_gain:
                best_info_gain = _score
                best_dim = _dim
                best_thresh = _thresh

    return best_dim, best_thresh, best_info_gain


def plot_labeled_data(x, y):
    """
    Plots the labeled data set
    """
    info_gain = information_gain(y, np.ones(len(y), dtype=bool))
    plt.figure()
    class_0 = y == 0
    class_1 = y == 1
    plt.scatter(x[class_0, 0], x[class_0, 1], marker='o', label='class 0')
    plt.scatter(x[class_1, 0], x[class_1, 1], marker='s', label='class 1')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.legend()
    plt.title('labeled data')
    plt.tight_layout()
    plt.savefig('labeled_data.png')
    plt.show()


def plot_decision_boundary(x, idx_split):
    """
    Plots the decision boundary of a binary classifier
    """
    idx_split = idx_split.astype(bool)
    plt.figure()
    plt.plot(x[idx_split, 0], x[idx_split, 1], 'r.', label='left child')
    plt.plot(x[~idx_split, 0], x[~idx_split, 1], 'b.', label='right child')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.legend()
    plt.title('decision boundary')
    plt.tight_layout()
    plt.savefig('decision_boundary.png')
    plt.show()


# main function to test the implementation
if __name__ == "__main__":

    # data set for the exercise (same as displayed in lecture on DBSCAN)
    data = np.loadtxt('decision_tree_dataset.txt', delimiter=',')
    x_train = data[:, :2]  # features
    y_train = data[:, -1].astype(int)  # targets

    # split the data w.r.t. maximum information gain
    best_dim, best_thresh, best_info_gain = best_split(x=x_train, y=y_train)
    print(f'best split (information gain={best_info_gain:.4f}) \n \
          obtained for: dim={best_dim}, \n \
          splitting value={best_thresh:.4f}')

    fig = plt.figure()
    class_0 = y_train == 0
    class_1 = y_train == 1
    plt.scatter(x_train[class_0, 0], x_train[class_0, 1], marker='o', label='class 0')
    plt.scatter(x_train[class_1, 0], x_train[class_1, 1], marker='s', label='class 1')
    if best_dim == 0:
        plt.axvline(x=best_thresh, color='black', label='best split')
    else:
        plt.axhline(y=best_thresh, color='black', label='best split')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.title(f'informatoin gain={best_info_gain:.4f}')
    plt.legend()
    plt.show()

    # draw random points from the range of the training data
    x_query = np.random.uniform(low=x_train.min(axis=0), 
                                high=x_train.max(axis=0), 
                                size=(1000, 2))
    
    # split the dataset according to the best split
    idx_split = create_split(x=x_query, split_dim=best_dim, 
                             split_val=best_thresh)
    
    # plot decision boundary
    plot_decision_boundary(x=x_query, idx_split=idx_split)


    """
    Task 3: Validate against sklearn
    """

    from sklearn.tree import DecisionTreeClassifier

    # fit a DT of depth 1
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(x_train, y_train)
  
    # predict the class labels of the query data
    y_pred = clf.predict(x_query)
    print(y_pred)
    
    # plot the decision boundary
    plot_decision_boundary(x=x_query, idx_split=y_pred)

    print(f'sklearn decision tree classifier: \n \
        best split (information gain={clf.tree_.impurity[0]:.4f}) \n \
        obtained for: dim={clf.tree_.feature[0]}, \n \
        splitting value={clf.tree_.threshold[0]:.4f}')
    
    # fit a DT of depth 5
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(x_train, y_train)

    # predict the class labels of the query data
    y_pred = clf.predict(x_query)

    plot_decision_boundary(x=x_query, idx_split=y_pred)



