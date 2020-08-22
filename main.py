import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import SimpleLogisticRegression
from theta_generator import GradientDescent, ThetaGenerator
from normalize import NoNormalize
from os import sys

if __name__ == "__main__":
    print('LOADING DATA...', end='')
    dataset = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)
    print('COMPLETE')

    print('INITALISE MACHINE', end='')
    machine = SimpleLogisticRegression(dataset, NoNormalize(), GradientDescent(10, 500))
    print('COMPLETE')
    print('LEARNING...', end='')
    machine.learn()
    print('COMPLETE')
    
    theta = machine.theta
    print('THETA = ' + str(theta))
    print('COMPUTE COST = %.2f' % ThetaGenerator.cost(machine.input_set, machine.output_set, theta), end='')
    print('(ALPHA = %.2f, NUMBER OF ITERATOR = %.2f)' %(machine.theta_generator.alpha, machine.theta_generator.iterator))

    loan = np.array(list(filter(lambda element: element[-1] == 1, dataset)))
    no_loan = np.array(list(filter(lambda element: element[-1] == 0, dataset)))

    y = (- theta[0, 0] - theta[1, 0] * machine.input_set[:, 1]) / theta[2, 0]
    
    print('DISPLAY PLOT:')
    print('\tFIGURE 1: DISPLAY RESULT')
    plt.figure(1)
    plt.plot(loan[:,0], loan[:,1], 'ro')
    plt.plot(no_loan[:,0], no_loan[:,1], 'bo', label='No')
    plt.legend(['Yes','No'])
    plt.plot(machine.input_set[:, 1], y, '-g')
    plt.xlabel('salary(1.000.000 VND)')
    plt.ylabel('experiences(years)')
    plt.suptitle('The posiblity of loan base on salary and experince')

    print('\tFIGURE 2: LEARNING PROGRESS')
    plt.figure(2)
    plt.plot(machine.theta_generator.cost_history[0], machine.theta_generator.cost_history[1], '-r')
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.suptitle('Learning progress')
    plt.show()
