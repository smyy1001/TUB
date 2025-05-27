import numpy as np


# Functional Approach
class_map = {}
inverse_class_map = {}

# --- Global dictionaries ---
label_to_index = {}
index_to_label = {}


# --- Functional Encoding Functions ---
def fit(data):
    global label_to_index, index_to_label
    unique_labels = sorted(set(data))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}


def encode(data):
    return [make_one_hot(label_to_index[label], len(label_to_index)) for label in data]


def decode(one_hot_list):
    return [index_to_label[np.argmax(one_hot)] for one_hot in one_hot_list]


def make_one_hot(index, size):
    one_hot = [0] * size
    one_hot[index] = 1
    return one_hot


# --- Load csv file ---
def load_data(filepath):
    return np.genfromtxt(filepath, dtype=str, delimiter=",")


# --- main ---
if __name__ == "__main__":
    data = load_data(
        "/home/sentropy/Desktop/AML/Exercise 02-20250506/bearing_faults.csv"
    )

    fit(data)
    for label, index in label_to_index.items():
        print(f"Unique Labels: {label}")
    print("\nNumber of unique labels:", len(label_to_index))

    encoded = encode(data)
    decoded = decode(encoded)

    print("\nSample original labels: ", data[:5])
    print("Sample one-hot encoded: ", encoded[:5])
    print("Sample decoded labels : ", decoded[:5])
