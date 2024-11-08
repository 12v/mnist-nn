import numpy as np
import random
import utils

training_image_path = "data/train-images.idx3-ubyte"
training_images = utils.read_idx_images(training_image_path)

training_label_path = "data/train-labels.idx1-ubyte"
training_labels = utils.read_idx_labels(training_label_path)


test_image_path = "data/t10k-images.idx3-ubyte"
test_images = utils.read_idx_images(test_image_path)

test_label_path = "data/t10k-labels.idx1-ubyte"
test_labels = utils.read_idx_labels(test_label_path)


hidden_layer_size = 16
hidden_layer_count = 2
training_iterations = 1000
learning_rate = 0.01
mini_batch_size = 10
input_layer_size = training_images.shape[1]
output_layer_size = training_labels.shape[1]


layer_sizes = (
    [input_layer_size] + [hidden_layer_size] * hidden_layer_count + [output_layer_size]
)

weights = [
    np.random.randn(next_layer_size, previous_layer_size)
    for previous_layer_size, next_layer_size in zip(layer_sizes[:-1], layer_sizes[1:])
]
biases = [np.random.randn(layer_size, 1) for layer_size in layer_sizes[1:]]


def feed_forward(image, weights, biases):
    activations = [image]
    zs = []
    for weight, bias in zip(weights, biases):
        z = np.dot(weight, activations[-1]) + bias
        zs.append(z)
        activations.append(utils.sigmoid(z))
    return zs, activations


def test():
    correct = 0
    for image, label in zip(test_images, test_labels):
        _, activations = feed_forward(image, weights, biases)
        output = activations[-1]
        if np.argmax(output) == np.argmax(label):
            correct += 1

    print(f"Accuracy: {correct / len(test_images) * 100:.2f}%")


for iteration in range(training_iterations):
    training_data = list(zip(training_images, training_labels))
    random.shuffle(training_data)
    mini_batches = [
        training_data[k : k + 10] for k in range(0, len(training_data), mini_batch_size)
    ]

    for mini_batch in mini_batches:
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]

        for image, label in mini_batch:
            # z = w * a + b
            zs, activations = feed_forward(image, weights, biases)

            # Backpropagation
            # compute the gradient of the cost function
            # with respect to the weights and biases

            # dC/dw = dC/da * da/dz * dz/dw
            # dC/db = dC/da * da/dz * dz/db

            # C = (a - label)^2
            # dC/da = 2(a - label)

            # a = sigmoid(z)
            # da/dz = sigmoid_prime(z)

            # z = w * a + b
            # dz/dw = a
            # dz/db = 1

            # therefore
            # dC/dw = 2(a - label) * sigmoid_prime(z) * a
            # dC/db = 2(a - label) * sigmoid_prime(z)

            cost_derivative = activations[-1] - label

            delta = cost_derivative * utils.sigmoid_prime(zs[-1])

            nabla_b[-1] += delta
            nabla_w[-1] += np.dot(delta, activations[-2].T)

            for l in range(2, len(weights) + 1):
                z = zs[-l]
                sp = utils.sigmoid_prime(z)
                delta = np.dot(weights[-l + 1].T, delta) * sp
                nabla_b[-l] += delta
                nabla_w[-l] += np.dot(delta, activations[-l - 1].T)

        weights = [
            w - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(weights, nabla_w)
        ]
        biases = [
            b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(biases, nabla_b)
        ]

    if iteration % 10 == 0:
        print(f"Iteration {iteration}")
        test()
