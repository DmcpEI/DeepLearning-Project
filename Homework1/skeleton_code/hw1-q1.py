#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Compute scores
        scores = self.W.dot(x_i)
        # Predict the label
        y_hat = np.argmax(scores)

        # If prediction is incorrect
        if y_hat != y_i:
            # Update the weights for the true class (increase weights)
            self.W[y_i, :] += x_i
            # Update the weights for the predicted class (decrease weights)
            self.W[y_hat, :] -= x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Compute scores for all classes
        scores = self.W.dot(x_i)

        # Compute the probability scores according to the model
        label_scores = np.expand_dims(scores, axis = 1)

        # Compute the one-hot encoding of the true class
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Compute the label probabilities according to the model
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        # Compute the gradient of the loss w.r.t. the weights
        gradient = (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)

        # Add L2 regularization to the gradient: lambda * W
        if l2_penalty > 0:
            gradient -= l2_penalty * self.W

        # Update the weights with regularization
        self.W += learning_rate * gradient


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.

        # Hidden layer weights and biases
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))  # Input to hidden weights
        self.b1 = np.zeros(hidden_size)  # Hidden layer bias
        # Output layer weights and biases
        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))  # Hidden to output weights
        self.b2 = np.zeros(n_classes)  # Output layer bias

        # raise NotImplementedError # Q1.3 (a)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=0))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=0)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.

        # Hidden layer
        z1 = np.dot(X, self.W1.T) + self.b1
        h1 = self.relu(z1)

        # Output layer
        z2 = np.dot(h1, self.W2.T) + self.b2
        y_hat = self.softmax(z2.T).T

        return np.argmax(y_hat, axis=1)  # Predicted class labels

        # raise NotImplementedError # Q1.3 (a)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """

        n_examples = X.shape[0]
        total_loss = 0

        for i in range(n_examples):
            x_i = X[i]  # Single example
            y_i = y[i]  # Single label

            # Forward pass
            z1 = np.dot(self.W1, x_i) + self.b1
            h1 = self.relu(z1)
            z2 = np.dot(self.W2, h1) + self.b2
            y_hat = self.softmax(z2)

            # Compute loss (cross-entropy)
            epsilon = 1e-6  # Small positive value to prevent log(0)
            total_loss += -np.log(y_hat[y_i] + epsilon)

            # Backward pass
            # Gradients for output layer
            dz2 = y_hat
            dz2[y_i] -= 1  # Subtract 1 from the true class probability
            dW2 = np.outer(dz2, h1)
            db2 = dz2

            # Gradients for hidden layer
            dh1 = np.dot(self.W2.T, dz2)
            dz1 = dh1 * self.relu_derivative(z1)
            dW1 = np.outer(dz1, x_i)
            db1 = dz1

            # Update weights and biases
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

        # Average loss over all examples
        return total_loss / n_examples

        # raise NotImplementedError # Q1.3 (a)


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='../data/intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Calculate final accuracies
    final_train_acc = train_accs[-1]
    final_val_acc = valid_accs[-1]

    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final train acc: {:.4f} | Final val acc: {:.4f}'.format(
        final_train_acc, final_val_acc
    ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
    ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final train acc: {final_train_acc}\n")
        f.write(f"Final val acc: {final_val_acc}\n")
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")

if __name__ == '__main__':
    main()
