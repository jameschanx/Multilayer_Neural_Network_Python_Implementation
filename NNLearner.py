"""
Neural Network Learner.  (c) 2018 James Chan
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class NNLearner(object):
    def __init__(self, learning_rate = .001,\
                       num_iterations = 100,\
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
        z = np.random.random(z.shape)
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_derivative(self,z):
        return self.sigmoid(z) * (1.0 * self.sigmoid(z))
    
    
    def display_params(self, weights, bias):
        pass
#        for W,b in zip(weights, bias):
#            for Wrow in W[:]:
#                print Wrow
#            for brow in b[:]:
#                print brow
            
        
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
            batch_samples = np.random.randint(0,num_examples - 1, self.batch_size)
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
            
#            if self.verbose:
#                if t % 100000 == 0:
#                    print "cost at time {}, is {}, learning_rate is {}".format(t, self.calculate_cost(A[len(A)-1], Y), self.learning_rate)
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
        return a.T

if __name__=="__main__":
    
    pass
#    train_X = np.array([[-3,-1,2,3,0,0,1,1,-4,3,-2,-1,2,0],
#                        [0,1,2,3,1,1,4,4,2,4,3,3,4,0]]).T
#    train_Y = np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,1]]).T
                      
#    train_X = np.array([[ 38.47, 38.2,  37.33, 37.71, 36.21, 36.72, 37.7, 37.27]]).T
#    train_Y = np.array([[ 0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]]).T      

#    train_X = np.array([[ 0 , 20, 30, 40, 50 ]]).T
#    train_Y = np.array([[ 0.,  0., 1., 1., 1.]]).T      
                      
#    train_X = np.array([[ 39 , 39, 41, 41, 41 ]]).T
#    train_X = (train_X - np.min(train_X))/(np.max(train_X) - np.min(train_X))
#    train_Y = np.array([[ 0.,  0., 1., 1., 1.]]).T   

#    train_X = np.array([[ 38.47, 38.2,  37.33, 37.71, 36.21, 36.72, 37.7, 37.27]]).T
#    train_X = (train_X - np.min(train_X))/(np.max(train_X) - np.min(train_X))
#    train_Y = np.array([[ 0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]]).T    
#                      
#    train_X = np.array([[ 38.47],
#     [ 38.2 ],
#     [ 37.33],
#     [ 37.71],
#     [ 36.21],
#     [ 36.72],
#     [ 37.7 ],
#     [ 37.27],
#     [ 37.73],
#     [ 35.73],
#     [ 37.79],
#     [ 36.52],
#     [ 36.11],
#     [ 37.27],
#     [ 41.7 ],
#     [ 41.01],
#     [ 39.81],
#     [ 41.57],
#     [ 43.28],
#     [ 43.19],
#     [ 43.24],
#     [ 44.01],
#     [ 42.16],
#     [ 40.03],
#     [ 39.88],
#     [ 41.15],
#     [ 39.97],
#     [ 39.54],
#     [ 39.51],
#     [ 40.23],
#     [ 38.87],
#     [ 39.45],
#     [ 39.07],
#     [ 39.36],
#     [ 39.29],
#     [ 40.07],
#     [ 40.16],
#     [ 39.88],
#     [ 40.51]])
#    #train_X = (train_X - np.min(train_X))/(np.max(train_X) - np.min(train_X))
#    train_Y = np.array([[ 0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  1.,
#      0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,
#      0.,  1.,  0.]]).T
#    
##   alpha = .001, epochs = 2000000, [3,3] layers
#    train_X = train_X[:20]
#    train_Y = train_Y[:20]
#
##    train_X = train_X[:5]
##    train_Y = train_Y[:5]
#
#    train_X = (train_X - np.min(train_X))/(np.max(train_X) - np.min(train_X))
#    for i in np.concatenate((train_X, train_Y), axis=1)[:]:
#        print 
#    
#    learning_rate = .0005
#    num_iterations = 2000000
#    num_neurons = [6,6]
#    batch_size = 20
#    
#    st = time.time()
#    nnl = NNLearner(learning_rate = learning_rate, 
#                     num_iterations = num_iterations, 
#                     num_neurons = num_neurons, 
#                     batch_size = batch_size, 
#                     verbose = True)
#    
#    nnl.addEvidence(train_X, train_Y)
#    print("training took: {} ms".format(time.time() - st))
#    prediction = dnnl.query(train_X)
#    print train_X
#    for i in range(prediction.shape[0]):
#        print prediction[i,:], train_Y[i,:]
#    
#    
    