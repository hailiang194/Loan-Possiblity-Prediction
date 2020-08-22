import numpy as np

class ThetaGenerator(object):
    @staticmethod
    def predict(predict_set, theta):
        '''
            predict the set
            @param predict_set set that needs to be predicted
            @param theta theta vector
            @return predict_set
        '''
        h = predict_set @ theta

        return 1 / (1 + np.exp(-h))
    
    @staticmethod
    def cost(X, y, theta):
        '''
            compute the cost of using theta as parameter for logistic regression to fit the data point in x and y
            @param X input set
            @param Y output set
            @param theta theta vector
            @return the cost
        '''
        predicted = ThetaGenerator.predict(X, theta)
        
        error = predicted - y

        sqr_error = np.transpose(error) @ error

        return (1 / (2 * np.size(y))) * sum(sqr_error)

    @staticmethod
    def cost_vector(X, y, theta):
        '''
            compute the cost of using theta as parameter for logistic regression to fit the data point in x and y
            @param X input set
            @param y output set
            @param theta theta vector
            @return the cost of each pair
        '''
        predicted = ThetaGenerator.predict(X, theta)

        return - y * np.log(predicted) - (1 - y) * (np.log(1 - predicted))

    def generate(self, X, y):
        '''
            generate theta vecot
            @param X input set
            @param Y output set
            @return theta vector
        '''
        pass

class GradientDescent(ThetaGenerator):
    def __init__(self, alpha, iterator):
        self.__alpha = alpha
        self.__iterator = iterator
        self.__cost_history = None

    @property
    def alpha(self):
        return self.__alpha

    @property
    def iterator(self):
        return self.__iterator

    @property
    def cost_history(self):
        return self.__cost_history

    def generate(self, X, y):
        theta = np.zeros((np.size(X, 1), 1))

        self.__cost_history = np.zeros((self.__iterator, 2))

        for time in range(self.__iterator):
            error = ThetaGenerator.predict(X, theta) - y
            theta = theta - (self.__alpha / np.size(y)) * (np.transpose(X) @ error)

            self.__cost_history[time, 0] = time
            self.__cost_history[time, 1] = ThetaGenerator.cost(X, y, theta)

        return theta
