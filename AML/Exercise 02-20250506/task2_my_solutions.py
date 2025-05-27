import numpy as np


class MyOneHotEncoder:
    def __init__(self):
        self._label_to_index = {}
        self._index_to_label = {}

    def fit(self, data):
        unique_labels = sorted(set(data))
        self._label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self._index_to_label = {i: label for label, i in self._label_to_index.items()}

    def encode(self, data):
        size = len(self._label_to_index)
        return [self._make_one_hot(self._label_to_index[label], size) for label in data]

    def decode(self, one_hot_list):
        return [self._index_to_label[np.argmax(vec)] for vec in one_hot_list]

    def _make_one_hot(self, index, size):
        vec = [0] * size
        vec[index] = 1
        return vec



if __name__ == "__main__":
    data = np.genfromtxt(
        "/home/sentropy/Desktop/summer'25/AML/Exercise 02-20250506/bearing_faults.csv",
        dtype=str,
        delimiter=",",
    )

    encoder = MyOneHotEncoder()
    encoder.fit(data)
    
    for label, index in encoder._label_to_index.items():
        print(f"Unique Labels: {label}")

    print("\nNumber of unique labels:", len(encoder._label_to_index))

    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)

    print("Sample original labels: ", data[:5])
    print("Sample one-hot encoded: ", encoded[:5])
    print("Sample decoded labels : ", decoded[:5])