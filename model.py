import numpy as np

from collections import OrderedDict

from sklearn.datasets import make_gaussian_quantiles
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class FeedForwardNW:
    def __init__(self, layer_config=None, activation=None, bias=True):
        self.layer_config = layer_config
        self.activation = activation
        self.bias = bias
        self.params = OrderedDict()
        self.grad = OrderedDict()

        self.v_dw = OrderedDict()
        self.s_dw = OrderedDict()
        self.v_db = OrderedDict()
        self.s_db = OrderedDict()

        self.activation_cache = OrderedDict()
        self.init_layers()

    def add_layer(self, layer_name, inp_len, out_len, bias=True, activation='relu'):
        self.params[layer_name] = {}
        self.grad[layer_name] = {}

        self.params[layer_name]['weights'] = np.random.randn(inp_len, out_len)
        self.grad[layer_name]['weights'] = np.zeros((inp_len, out_len))
        self.params[layer_name]['activation'] = activation

        # Adam prop initialization
        self.v_dw[layer_name] = np.zeros((inp_len, out_len))
        self.s_dw[layer_name] = np.zeros((inp_len, out_len))
        if self.bias:
            self.params[layer_name]['bias'] = np.random.randn(1, out_len)
            self.grad[layer_name]['bias'] = np.zeros((1, out_len))

            # Adam prop initialization
            self.v_db[layer_name] = np.zeros((1, out_len))
            self.s_db[layer_name] = np.zeros((1, out_len))

    def relu(self, Z):
        return np.maximum(Z, 0, Z)

    def init_layers(self):
        if self.layer_config:
            for i in range(len(self.layer_config)-1):
                self.add_layer(
                    'layer_'+str(i+1), self.layer_config[i], self.layer_config[i+1], activation=self.activation[i])

        else:
            return {}

    def layer_out(self, A, layer_name, with_grad):
        out = np.dot(A, self.params[layer_name]['weights']
                     ) + self.params[layer_name]['bias']
        if self.params[layer_name]['activation'] == 'relu':
            out = self.relu(out)
        elif self.params[layer_name]['activation'] == 'softmax':
            out = softmax(out, axis=1)

        if with_grad:
            self.activation_cache[layer_name] = out

        return out

    def forward_prop(self, X, with_grad=False):
        self.activation_cache['layer_0'] = X
        out = X
        for layer_name in self.params:
            out = self.layer_out(out, layer_name, with_grad)
        return out

    def loss(self, Y, A):
        return np.sum(-Y * np.log(A+1e-8))/len(Y)
    
    def accuracy(self, Y, A):
        y_hat = np.argmax(A, axis = 1)
        actual = np.argmax(Y, axis = 1)
        return np.mean(y_hat == actual)

    def compute_rms_momentum(self, layer_name, grad_weights, grad_bias, beta1, beta2, lambda_, batch_size):
        grad_weights += ((lambda_/batch_size) * self.params[layer_name]['weights'])
        # momentum
        self.v_dw[layer_name] = (beta1 *
                                 self.v_dw[layer_name]) + ((1-beta1) * grad_weights)
        self.v_db[layer_name] = (
            beta1*self.v_db[layer_name]) + ((1-beta1) * grad_bias)

        # rms
        self.s_dw[layer_name] = (beta2*self.s_dw[layer_name]) + \
            ((1-beta2) * np.power(grad_weights, 2))
        self.s_db[layer_name] = (beta2*self.s_db[layer_name]) + \
            ((1-beta2) * np.power(grad_bias, 2))


    def compute_batch_gradients(self, X, Y, lambda_ ,beta1=0.9, beta2=0.999):
        n = len(Y)
        layer_names = list(self.activation_cache.keys())
        final_layer = layer_names.pop(-1)
        A = self.activation_cache[final_layer]
        dZ_prev = A-Y
        W_prev = self.params[final_layer]['weights']
        grad_weights = np.dot(
            self.activation_cache[layer_names[-1]].T, dZ_prev)/n
        grad_bias = np.sum(dZ_prev, axis=0, keepdims=True)/n
        self.compute_rms_momentum(
            final_layer, grad_weights, grad_bias, beta1, beta2, lambda_, n)

        layer_len = len(layer_names)

        for i, layer_name in enumerate(layer_names[1:][::-1]):
            activation_grad = (self.activation_cache[layer_name] > 0) * 1
            dZ_prev = (np.dot(dZ_prev, W_prev.T)) * activation_grad
            W_prev = self.params[layer_name]['weights']
            grad_weights = np.dot(
                self.activation_cache[layer_names[layer_len-i-2]].T, dZ_prev)/n
            grad_bias = np.sum(dZ_prev, axis=0, keepdims=True)/n
            self.compute_rms_momentum(
                layer_name, grad_weights, grad_bias, beta1, beta2, lambda_, n)


    def update_params(self, lr, epoch, beta1=0.9, beta2=0.99):
        for layer_name in self.params:
            # bias correction
            v_dw = self.v_dw[layer_name] / (1-beta1 ** epoch)
            v_db = self.v_db[layer_name] / (1-beta1 ** epoch)

            s_dw = self.s_dw[layer_name]/(1-beta2 ** epoch)
            s_db = self.s_db[layer_name]/(1-beta2 ** epoch)
           

            self.params[layer_name]['weights'] -= lr *(v_dw /(np.power(s_dw, 0.5) + 1e-8))
            self.params[layer_name]['bias'] -= lr * (v_db / (np.power(s_db, 0.5) + 1e-8))


    def batches(self, X, y, batch_size, shuffle):
        if shuffle:
            assert len(X) == len(y)
            p = np.random.permutation(len(y))
            X = X[p]
            y = y[p]
        num_batches = int(len(y)/batch_size)
        for i in range(num_batches):
            start_ind = i * batch_size
            stop_ind = start_ind + batch_size
            yield (X[start_ind:stop_ind], y[start_ind:stop_ind])

        n = num_batches * batch_size
        if (n) != len(y):
            yield (X[n-1:], y[n-1:])

    def train(self, X_train, y_train, X_test, y_test, batch_size, num_epochs, lr, lambda_= 0, shuffle=True, verbose=True):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(num_epochs):
            loss = []
            accuracy = []
            
            for batch in self.batches(X_train, y_train, batch_size, shuffle):
                A = self.forward_prop(batch[0], with_grad=True)
                loss.append(self.loss(batch[1], A))
                accuracy.append(self.accuracy(batch[1], A))
                self.compute_batch_gradients(batch[0], batch[1], lambda_ = lambda_)
                self.update_params(lr, epoch+1)
            
            A_test = self.forward_prop(X_test, with_grad=False)
            test_loss = self.loss(y_test, A_test)
            test_accuracy = self.accuracy(y_test, A_test)
            
            train_loss = np.mean(loss)
            train_accuracy = round(np.mean(accuracy), 2)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            
            if verbose:
                print('Epoch {}  Train loss: {}, Test loss: {}, Train accuracy: {}, Test Accuracy: {}'.format(
                    epoch+1, train_loss, test_loss, train_accuracy, test_accuracy))
            
        return train_losses, test_losses, train_accuracies, test_accuracies


def plot_train_test_metric(train_history, test_history, num_epochs):
    epochs = range(1,num_epochs+1)
    plt.plot(epochs, train_history, 'g', label='Training loss')
    plt.plot(epochs, test_history, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()