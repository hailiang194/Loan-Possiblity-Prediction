import numpy as np
from theta_generator import ThetaGenerator

class SimpleLogisticRegression(object):
    def __init__(self, dataset, normalizer, theta_generator):
        self.__dataset = dataset
        self.__normalizer = normalizer
        self.__theta_generator = theta_generator
        self.__theta = np.zeros((np.size(self.__dataset, 1), 1))

    def __setup_training_set(self):
        '''
            get input and output pair for learning
            @return (input_set, output_set)
        '''
        X = np.copy(self.__dataset)
        y = np.zeros((np.size(self.__dataset, 0), 1))

        X[:, 0] = 1
        X[:,1:] = self.__dataset[:,:-1]

        y[:, 0] = self.__dataset[:,-1]

        return X, y

    @property
    def theta(self):
        return self.__theta

    @property
    def input_set(self):
        return self.__setup_training_set()[0]

    @property
    def output_set(self):
        return self.__setup_training_set()[1]

    @property
    def theta_generator(self):
        return self.__theta_generator

    def learn_more(self, dataset):
        '''
            Append new data sets and learn again
            @param dataset appended dataset
        '''
        self.__dataset = np.append(self.__dataset, dataset, axis=0)
        self.learn()

    def learn(self):
        '''
            Generate theta vector
        '''
        X, y = self.__setup_training_set()
        self.__theta = self.__theta_generator.generate(self.__normalizer.normalize_input(X), y)

    def predict(self, predict_set):
        '''
            Predict the set
            Note: Each row of predict_set must be start with 1
            @param predict_set set that need to be predicted
            @return predicted set
        '''
        return ThetaGenerator.predict(self.__normalizer.normalize_predict(predict_set), self.__theta)
