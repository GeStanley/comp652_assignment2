__author__ = 'geoffrey'

import numpy
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def five_fold_cross_validation(feature_array, target_vector):

    complete_array = numpy.append(feature_array, target_vector[:, numpy.newaxis], axis=1)

    complete_array = shuffle(complete_array, random_state=125)

    five_folds = numpy.array_split(complete_array, 5)

    # go through the 5 training cases
    for i in range(0, 5):
        training_data = numpy.empty((0, complete_array.shape[1]), dtype=float)
        testing_data = numpy.empty((0, complete_array.shape[1]), dtype=float)

        for j in range(0, 5):
            if j == i:
                testing_data = numpy.append(testing_data, five_folds[j], axis=0)
            else:
                training_data = numpy.append(training_data, five_folds[j], axis=0)


        # split the training data into features and targets
        training_features = training_data[:, 0:training_data.shape[1]-1]
        training_target = training_data[:, training_data.shape[1]-1]

        logReg = LogisticRegression()
        logReg.fit(training_features, training_target)

        # split the testing data into features and targets
        testing_features = testing_data[:, 0:testing_data.shape[1]-1]
        testing_target = testing_data[:, testing_data.shape[1]-1]

        print logReg.predict_proba(testing_features)
        print logReg.predict(testing_features)
        print testing_target
        print

def calculate_sigmoid_hypothesis(weights, feature_vector):

    exponent = numpy.dot(weights.T, feature_vector)

    return 1/(1+math.exp(exponent))

def calculate_sigmoid_hypothesis_vector(weights, feature_array):

    vector = numpy.empty(0, dtype=float)

    for i in range(0, feature_array.shape[0]):
        hypothesis = calculate_sigmoid_hypothesis(weights, feature_array[i])

        vector = numpy.append(vector, [hypothesis], axis=0)


    return vector

def calculate_loss(feature_array, target_vector, weights):

    hypothesis_vector = calculate_sigmoid_hypothesis_vector(weights, feature_array)

    error = hypothesis_vector - target_vector
    gradient = numpy.dot(feature_array.T, error)

    return gradient

def logistic_regression(feature_array, target_vector, learning_rate=0.001, precision=0.000001):

    weights = numpy.ones(feature_array.shape[1], dtype=float)

    step = learning_rate * calculate_loss(feature_array, target_vector, weights)

    while abs(numpy.sum(step)) > precision:

        weights += step
        step = learning_rate * calculate_loss(feature_array, target_vector, weights)


    return weights

def covariance(vector_a, vector_b):

    N = vector_a.shape[1] - 1

    a_avg = numpy.mean(vector_a)
    b_avg = numpy.mean(vector_b)

    error_a = vector_a - a_avg
    error_b = vector_b - b_avg

    return numpy.dot(error_a, error_b.T) / N

def gaussian_discriminant_analysis(feature_array, target_vector):
    #p(x |  y = 0 ) = 1/(2*pie)

    data_set = numpy.append(feature_array, target_vector[:, numpy.newaxis], axis=1)

    model_output = numpy.empty([feature_array.shape[0], 2], dtype=float)

    cov_matrix = 0

    for y in range(0, 2):
        sub_set = data_set[data_set[:, data_set.shape[1] - 1] == y]

        mean = numpy.mean(sub_set, axis=0)

        for i in range(0, data_set.shape[0]-1):
            row = data_set[:, i]




if __name__ == '__main__':

    array_x = numpy.loadtxt('wpbcx.dat', float)
    vector_y = numpy.loadtxt('wpbcy.dat', float)

    # gaussian_discriminant_analysis(array_x, vector_y)
    #
    plt.scatter(array_x[:, 10], array_x[:, 2], c=vector_y)
    plt.show()


    #five_fold_cross_validation(array_x, vector_y)


    weights = logistic_regression(array_x, vector_y)


    logReg = LogisticRegression()
    logReg.fit(array_x, vector_y)

    print weights
    print logReg.coef_
    # for i in range(0, array_x.shape[1]):
    #     print calculate_sigmoid_hypothesis(weights, array_x[i,])

   # print logReg.predict_proba(array_x)

    # print array_x.shape
    #
    # logReg = LogisticRegression()
    # logReg.fit(array_x, vector_y)
    #
    # log_probability = logReg.predict_log_proba(array_x)
    #
    # print logReg.coef_.shape
    # print log_probability.shape
    # print log_probability