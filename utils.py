import numpy as np


def read_idx_images(filename):
    with open(filename, "rb") as f:
        # Read the magic number
        int.from_bytes(f.read(4), byteorder="big")
        # Read the number of images
        num_images = int.from_bytes(f.read(4), byteorder="big")
        # Read the number of rows
        num_rows = int.from_bytes(f.read(4), byteorder="big")
        # Read the number of columns
        num_cols = int.from_bytes(f.read(4), byteorder="big")

        # Read the image data
        image_data = f.read(num_images * num_rows * num_cols)

        images = np.frombuffer(image_data, dtype=np.uint8).reshape(
            num_images, num_rows * num_cols, 1
        )

        images = images / 255.0

        return images


def read_idx_labels(filename):
    with open(filename, "rb") as f:
        # Read the magic number
        int.from_bytes(f.read(4), byteorder="big")
        # Read the number of labels
        num_labels = int.from_bytes(f.read(4), byteorder="big")

        # Read the label data
        label_data = f.read(num_labels)

        labels = np.frombuffer(label_data, dtype=np.uint8)

        num_classes = len(np.unique(labels))

        one_hot_labels = []
        for label in labels:
            one_hot = np.zeros((num_classes, 1))
            one_hot[label] = 1
            one_hot_labels.append(one_hot)

        return np.array(one_hot_labels)


def softmax(z):
    e_x = np.exp(z - np.max(z))
    return e_x / e_x.sum(axis=0)


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
