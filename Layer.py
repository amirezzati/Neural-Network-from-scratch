import numpy as np
from sklearn.model_selection import train_test_split


# X = [ [ |,  |,  |,  |,   |],
#       [x1, x2, x3, ..., xm],
#       [ |,  |,  |,  |,   |]  ]

class Layer():
    def __init__(self, input_num, nuerons_num):
        # guassian distributation with 0 mean and standard deviation equals to inputs number
        self.w = np.random.default_rng().normal(
            0, 1/np.sqrt(input_num), (nuerons_num, input_num))
        self.w = np.random.random((nuerons_num, input_num)) - 0.5
        self.b = np.zeros((nuerons_num, 1))

    def forward_prop(self, input):
        self.x = input
        self.Z = np.dot(self.w, self.x) + self.b

    def back_prop(self, m, y_actual, dz_next=0, w_next=0):
        if type(dz_next) == int or type(w_next) == int:  # calculate dz for output layer
            self.dz = self.A - y_actual
        else:  # calculate dz for other layers
            # dz[l-1] = w[l].T dz[l] * g[l-1]'(z[l-1])
            self.dz = np.dot(w_next.T, dz_next) * (self.A * (1-self.A))

        self.dw = 1/m * np.dot(self.dz, self.x.T)
        self.db = 1/m * np.sum(self.dz, axis=1, keepdims=True)

    def update_wb(self, alpha):
        self.w = self.w - alpha * self.dw
        self.b = self.b - alpha * self.db

    def sigmoid_activation(self):
        self.A = 1 / (1 + np.exp(-self.Z))


def calculate_cost(y_hat, y):
    y_hat_clipped = np.clip(y_hat, pow(10, -10), 1-pow(10, -10))
    loss = - np.sum(y * np.log(y_hat_clipped) + (1-y) *
                    np.log(1-y_hat_clipped), axis=0, keepdims=True)
    cost = np.mean(loss)
    return cost


def convert_to_binary(n, threshold):
    for i in range(len(n)):
        n[i] = [1 if x > threshold else 0 for x in n[i]]
    return n


def accuracy(y_hat, y):
    y_hat_rounded = convert_to_binary(y_hat, 0.9)
    subtract = y_hat_rounded - y
    count = np.count_nonzero(subtract == 0)
    return count / y.shape[1]
