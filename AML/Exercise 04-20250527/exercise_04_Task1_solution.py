"""
Solution to task 1 of Exercise 4: Implementation of a Zscoring class 

(C) Merten Stender, TU Berlin, 2025
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import os


class ZScorer:

    def __init__(self):
        self._means: np.ndarray
        self._stds: np.ndarray

    def fit(self, data: np.ndarray):
        # Fits the class object to the data set,
        # extracts mean and std per feature column

        self._means = np.mean(data, axis=0)
        self._stds = np.std(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        # returns a zscored data set

        # 1. subtract the mean from the data set, columnwise
        data_transformed = data - self._means

        # 2. devide by standart deviations (columnwise)
        data_transformed = data_transformed / self._stds

        return data_transformed
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # reverse the zscoring transformation

        # 1. multiply with std
        data_reversed = data * self._stds

        # 2. add the means
        data_reversed = data_reversed + self._means
        
        return data_reversed


if __name__ == "__main__":

    # Load example data for clustering (code given in assignment)
    data_path = os.path.join(os.path.dirname(__file__), 'data_clustering.csv')
    data = np.loadtxt(data_path, delimiter=',')

    # Print the shape of the data, and the mean and standard deviation 
    # per column
    print(f"\n\nData shape: {data.shape}")
    print(f"Mean of data: {np.mean(data, axis=0)}")
    print(f"Std. of data: {np.std(data, axis=0)}")

    # TASK 1: Implement a Zscoring class.

    # 1. Instantiate a zscorer object
    scaler = ZScorer()  # your own implementation
    # scaler = StandardScaler() # sklearn  <--- very last task of the assignment

    # 2. Fit the scaler to the data
    scaler.fit(data)

    # 3. Transform the data using the fitted scaler
    data_zscored = scaler.transform(data)

    # 4. Print the mean and standard deviation of the transformed data
    print(f"Mean of transformed data: {np.mean(data_zscored, axis=0)}")
    print(f"Std. of transformed data: {np.std(data_zscored, axis=0)}")

    # # 5. Revert the transformation
    data_reverted = scaler.inverse_transform(data_zscored)

    # 6. Print the mean and standard deviation of the reverted data
    print(f"Mean of reverted data: {np.mean(data_reverted, axis=0)}")
    print(f"Std. of reverted data: {np.std(data_reverted, axis=0)}")

    # 7. Check if the reverted data is close to the original data
    print(f"Are the reverted data and original data close? \
          {np.allclose(data, data_reverted)}")









    