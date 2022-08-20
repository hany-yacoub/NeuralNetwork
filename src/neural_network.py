import numpy as np
from .activation import tanh, softmax

class neural_network:
    def __init__(self,num_hid):
        # initializing weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3

        # records number of epochs without improvement, the condition to stop.
        count = 0
        best_valid_acc = 0

        while count<=50:
            

            #Forward pass
            z_1 = tanh((train_x.dot(self.weight_1)+self.bias_1))
            y = softmax(z_1.dot(self.weight_2)+self.bias_2)


            # Backward pass (backpropagation)

            delta_v = np.dot(z_1.T, train_y - y)
            delta_vbias = train_y - y

            delta_w = np.dot(train_x.T, (1-z_1**2) * np.dot(train_y - y , self.weight_2.T))
            delta_wbias = np.dot(train_y - y , self.weight_2.T) * (1-z_1**2)

            #update weights

            self.weight_1 = self.weight_1 + lr * delta_w
            self.weight_2 = self.weight_2 + lr * delta_v
           
            self.bias_1 = self.bias_1 + lr * np.sum(delta_wbias, axis=0)
            self.bias_2 = self.bias_2 + lr * np.sum(delta_vbias, axis=0) 

            # validate
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # finds probability of each class

        z = tanh(x.dot(self.weight_1)+self.bias_1)
        y = softmax(z.dot(self.weight_2)+self.bias_2)

        # classifies based on probability

        labels = np.zeros([len(x),]).astype('int')

        for i in range(len(y)):
            labels[i] = np.argmax(y[i])

        return labels

    def get_hidden(self,x):
        # extract intermediate features
        z = tanh(x.dot(self.weight_1)+self.bias_1)

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
