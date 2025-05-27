"""
Solution to exercise 02: one-hot encoding

DISCLAIMER:
Please note that this solution may contain errors (please report them, thanks!),
and that there are most-likely more elegant and more efficient implementations
available for the given problem. In this light, this solution may only serve as
an inspiration for solving the given exercise tasks.

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""

import numpy as np

"""
TASK1: Functions for one-hot encoding
"""


def fit(data: np.ndarray | list) -> dict:
    """ Fit the encoder to the data.
    data: list or one-dimensional array of categorical data
    returns: dictionary with mapping between category (key) and index (value)
    """
    # obtain number of classes and classes themselves
    classes = np.unique(data)

    # mapping between category (key) and index (value) via dictionary
    # using an explicit for loop to create a dictionary key:value pair,
    class_map = {}
    for i, _class in enumerate(classes):
        class_map[_class] = i

    # # short pythonic way to create a dictionary
    # indices = range(num_classes)
    # class_map = dict(zip(classes, indices))

    return class_map


def encode(data: np.ndarray | list, class_map: dict) -> np.ndarray:
    """ One-hot encode categorical data.
    data: list or one-dimensional array of categories
    class_map: dictionary with mapping between category (key) and index (value)
    returns: binary N x K binary matrix. N samples, K categories
    """
    num_classes = len(class_map)

    # one-hot encode the categorical data 
    encoded_vals = [] 
    for val in data: 
        _enc_value = np.zeros(num_classes) # empty vector of zeros
        _enc_value[class_map[val]] = 1 # turn 'hot' (put in a one)
        encoded_vals.append(_enc_value) # stack to existing values

    return np.vstack(encoded_vals)  # turn into binary matrix and return


def decode(enc_vals: np.ndarray | list, class_map: dict) -> np.ndarray:
    """ Invert one-hot encoding.
    enc_vals: binary N x K binary matrix. N samples, K categories
    class_map: dictionary with mapping between category (key) and index (value)
    returns: list or one-dimensional array of categories
    """
    # inverse mapping between index and category
    inv_class_map = {v: k for k, v in class_map.items()}

    # de-code one-hot encoded values
    values = []
    for enc_val in enc_vals:

        idx = np.argwhere(enc_val == 1)[0][0]
        values.append(inv_class_map[idx])

    return np.hstack(values)  # return a one-dim. np.ndarray of categories


"""
TASK 2: Object-oriented implementation of one-hot encoding
"""
class OneHotEncoder():
    """K classes into K-dimensional binary vector.
    Will not take care of multicollinearity!
    """

    def __init__(self):
        self.classes: list = []
        self.num_classes: int = None
        self._class_map = None
        self._inv_class_map = None
        
    def fit(self, values):
        # obtain number of classes and classes themselves
        self.classes = np.unique(values)
        self.num_classes = len(self.classes)

        # mapping between category (key) and index (value) via dictionary
        indices = range(self.num_classes)
        self._class_map = dict(zip(self.classes, indices))
        pass

    def encode(self, values):
        """ One-hot encode categorical data.
        expects list or one-dimensional array of categories
        returns a binary N x K binary matrix. N samples, K categories
        """
        
        # Ensure the encoder is fitted
        if self._class_map is None:
            raise ValueError("The encoder has not been fitted yet. Please call 'fit' before 'encode'.")
        
        # one-hot encode the categorical data 
        encoded_vals = [] 
        for val in values: 
            _enc_value = np.zeros(self.num_classes) # empty vector of zeros
            _enc_value[self._class_map[val]] = 1 # turn 'hot' (put in a one)
            encoded_vals.append(_enc_value) # stack to existing values

        return np.vstack(encoded_vals)  # turn into binary matrix and return

    def decode(self, enc_vals):
        """ Invert one-hot encoding.
        if self._class_map is None:
            raise ValueError("The encoder has not been fitted yet. Please call 'fit' before decoding.")
        self._inv_class_map = {v: k for k, v in self._class_map.items()}
        returns list or one-dimensional array of categories
        """
        # inverse mapping between index and category
        self._inv_class_map = {v: k for k, v in self._class_map.items()}

        # de-code one-hot encoded values
        values = []
        for enc_val in enc_vals:

            idx = np.argwhere(enc_val == 1)[0][0]
            values.append(self._inv_class_map[idx])

        return np.hstack(values)  # return a one-dim. np.ndarray of categories


if __name__ == "__main__":

    # simple test data for one-hot encoding
    values = np.array(['Berlin', 'Frankfurt', 'Munich', 'Berlin'])
    print(f'values to encode: \t{values}')

    """
    Task 1: One-hot encoding via functions
    """
    print("\n --------- TASK 1 --------- \n")
    class_map = fit(values)
    print(f'class map: {class_map}')

    enc_values = encode(values, class_map)
    print(f'one-hot encoded representation: \n {enc_values}')

    dec_values = decode(enc_values, class_map)
    print(f'de-coded values: \t{dec_values}')

    """
    Task 2: One-hot encoding via object-oriented implementation
    """
    print("\n --------- TASK 2 --------- \n")
    # instantiate the encoder
    ohe_encoder = OneHotEncoder()

    # fit the encoder
    ohe_encoder.fit(values)
    print(f'class map: {ohe_encoder._class_map}')  # even w/ underscore, this is public

    # encode the data
    enc_values = ohe_encoder.encode(values)
    print(f'one-hot encoded representation: \n {enc_values}')

    # decode the data (i.e. after model prediction)
    dec_values = ohe_encoder.decode(enc_values)
    print(f'de-coded values: \t{dec_values}')

    # validate against scikit-learn
    print("\n --------- Sklearn Validation --------- \n")
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

    encoder_sklearn = SklearnOneHotEncoder(sparse_output=False)
    encoder_sklearn.fit(values.reshape(-1, 1))  # sklearn expects 2D array
    print(f'sklearn class map: {encoder_sklearn.categories_}')
    print(f'sklearn one-hot encoded representation: \n {encoder_sklearn.transform(values.reshape(-1, 1))}')
    print(f'sklearn de-coded values: \t{encoder_sklearn.inverse_transform(encoder_sklearn.transform(values.reshape(-1, 1)))}')



    """
    Bearing fault data: one-hot encoding
    """
    print("\n --------- Bearing data --------- \n")
    # load the data
    data = np.genfromtxt('bearing_faults.csv', dtype='str', delimiter=",")
    print(f'data sample: {data[:5]}')

    # one-hot encode data
    encoder = OneHotEncoder()
    encoder.fit(data)
    print(f'class map: {encoder._class_map}')

    # encode the data
    data_enc = encoder.encode(data)
    print(f'encoded data: {data_enc[:5]}')

    # decode the data (i.e. after model prediction)
    data_dec = encoder.decode(data_enc)
    print(f'decoded data: {data_dec[:5]}')


