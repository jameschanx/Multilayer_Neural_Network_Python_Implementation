"""
Neural Network Learner.  (c) 2018 James Chan
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class NNLearner(object):
    def __init__(self, learning_rate = .001,\
                       num_iterations = 500,\
                       num_neurons = [],\
                       batch_size = 32,\
                       verbose = True):
        
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        
        num_neurons.append(1) #append 1 here to make number of neuron at output layer be 1.
        self.num_neurons = num_neurons
        self.num_of_layers = len(self.num_neurons)
        self.batch_size = batch_size
        self.verbose = verbose
        
    def calculate_cost(self, yhat, y):
        return np.sum(-y * np.log(yhat) - (1-y) * np.log(1-yhat))/y.shape
    
    def calculate_derivative(self, dataY, a):
        dLdz = a - dataY
        dLdW = dLdz * a.T
        dLdb = dLdz
        return dLdW, dLdb
    
    def sigmoid(self, z):
#        z = np.random.random(z.shape)
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1.0 * self.sigmoid(z))
    
    
    def display_params(self, weights, bias):
        pass
        for W,b in zip(weights, bias):
            for Wrow in W[:]:
                print(Wrow)
            for brow in b[:]:
                print(brow)
            
        
    def addEvidence(self, dataX, dataY):
        dataX = dataX.T
        dataY = dataY.T
#        print dataX.shape
#        print dataY.shape
        num_examples = dataX.shape[1]
        if self.batch_size > num_examples:
            self.batch_size = num_examples
        
        #weight initialization
        W = []
        b = []
        for i in range(self.num_of_layers):
            if i == 0:
                num_inputs = dataX.shape[0]
            else:
                num_inputs = self.num_neurons[i-1]
            ith_layer_weights = (np.random.random((self.num_neurons[i], num_inputs))-.5)
            ith_layer_bias = np.zeros((self.num_neurons[i], 1))
            #ith_layer_bias = (np.random.random((self.num_neurons[i], 1))-.5)
            W.append(ith_layer_weights)
            b.append(ith_layer_bias)
        
        #commence training
        for t in range(self.num_iterations + 1):
            
            #forward pass
            batch_samples = np.random.randint(0, num_examples - 1, self.batch_size)
            A = [dataX[:, batch_samples]]
            Y = dataY[batch_samples]
            #A = [dataX]
            #Y = dataY
            for i in range(self.num_of_layers):
                z = np.dot(W[i], A[i]) + b[i]
                a = self.sigmoid(z)
                A.append(a)
            #backprop
            dCdW = []
            dCdb = []
            for i in reversed(range(1, self.num_of_layers + 1)):
                if i == (self.num_of_layers):
                    dCdz = A[i] - Y
                else:
                    z = np.dot(W[i - 1], A[i - 1]) + b[i - 1]
                    dCdz = np.dot(W[i].T, dCdz) * self.sigmoid(z) * (1 - self.sigmoid(z))
                dCdW.insert(0, np.dot(dCdz, A[i - 1].T) / num_examples)
                dCdb.insert(0, np.sum(dCdz, axis=1, keepdims=True) / num_examples)
            
            for i in range(self.num_of_layers):
                W[i] = W[i] - self.learning_rate * dCdW[i]
                b[i] = b[i] - self.learning_rate * dCdb[i]   
            
            if t % 1000 == 0:
                print("cost at time {}, is {}, learning_rate is {}"
                      .format(t, self.calculate_cost(A[len(A)-1], Y), self.learning_rate))
        self.W = W
        self.b = b
        
    def query(self,points):
        dataX = points.T
        a = dataX
        for i in range(self.num_of_layers):
            z = np.dot(self.W[i], a) + self.b[i]
            a = self.sigmoid(z)
            
        a[a > 0.5] = 1
        a[a <= 0.5] = 0
        return a.T[:,0]

if __name__=="__main__":
    seed = np.random.seed(56)
    n_samples = 1000
    n_features = 2
    centers = 2
    cluster_std = 3.0
    data = make_blobs(n_samples, n_features, centers, cluster_std)
    plt.figure(1)
    plt.scatter(data[0][:,0], data[0][:,1], c=data[1])
    
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2)
    plt.figure(2)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
    
    
    learning_rate = .001
    num_iterations = 50000
    num_neurons = [3]
#    num_neurons = []
    batch_size = 32
    
    nnl = NNLearner(learning_rate = learning_rate, 
                     num_iterations = num_iterations, 
                     num_neurons = num_neurons, 
                     batch_size = batch_size, 
                     verbose = True)
    
    nnl.addEvidence(X_train, y_train)
    predictions = nnl.query(X_train)
    print("accuracy = {}".format(sum(predictions == y_train)/y_train.shape[0]))
    