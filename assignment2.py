__author__ = 'geoffrey'

import numpy
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


def five_fold_cross_validation(feature_array, target_vector, model, cov_type='full'):

    complete_array = numpy.append(feature_array, target_vector[:, numpy.newaxis], axis=1)

    complete_array = shuffle(complete_array, random_state=125)

    five_folds = numpy.array_split(complete_array, 5)

    returned_statistics = {}
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

        # split the testing data into features and targets
        testing_features = testing_data[:, 0:testing_data.shape[1]-1]
        testing_target = testing_data[:, testing_data.shape[1]-1]

        if model == 'logistic':
            returned_statistics[i] = logistic_regression(training_features,
                                                         training_target,
                                                         testing_features,
                                                         testing_target)
        elif model == 'gaussian':
            returned_statistics[i] = gaussian_discriminant_analysis(training_features,
                                                                    training_target,
                                                                    testing_features,
                                                                    testing_target,
                                                                    cov_type)
        elif model == 'klr':
            returned_statistics[i] = kernelized_logistic_regression(training_features,
                                                                    training_target,
                                                                    testing_features,
                                                                    testing_target)


    return returned_statistics


def logistic_regression(training_features, training_targets, testing_features, testing_targets):

    logReg = LogisticRegression()
    logReg.fit(training_features, training_targets)

    training_average = average_example_log_likelihood(logReg.predict_log_proba(training_features),
                                                      training_targets)
    testing_average = average_example_log_likelihood(logReg.predict_log_proba(testing_features),
                                                     testing_targets)

    training_accurate = numpy.sum(logReg.predict(training_features) == training_targets)
    training_accuracy = float(training_accurate) / float(training_targets.size)

    testing_accurate = numpy.sum(logReg.predict(testing_features) == testing_targets)
    testing_accuracy = float(testing_accurate) / float(testing_targets.size)

    fold_data = {'log L train': training_average,
                 'log L test': testing_average,
                 'training accuracy': training_accuracy,
                 'testing accuracy': testing_accuracy}

    return fold_data


def average_example_log_likelihood(class_likelihoods, targets):

    log_likelihood_sum = 0

    m = class_likelihoods.shape[0]

    for i in range(0, m):

        log_likelihood_sum += class_likelihoods[i, targets[i]]

    return log_likelihood_sum / m



def gaussian_discriminant_analysis(training_features, training_targets, testing_features, testing_targets, cov_type='full'):

    train_data_set = numpy.append(training_features, training_targets[:, numpy.newaxis], axis=1)
    test_data_set = numpy.append(testing_features, testing_targets[:, numpy.newaxis], axis=1)

    train_log_likelihoods = numpy.empty([training_features.shape[0], 2], dtype=float)
    test_log_likelihoods = numpy.empty([testing_features.shape[0], 2], dtype=float)

    for y in range(0, 2):
        # subset the data into one of the two classes
        train_sub_set = train_data_set[train_data_set[:, train_data_set.shape[1] - 1] == y, :-1]

        prior = float(train_sub_set.shape[0]) / float(train_data_set.shape[0])
        mean = numpy.mean(train_sub_set, axis=0)

        if cov_type == 'full':
            covariance_matrix = numpy.cov(train_data_set[:, :-1].T)
        elif cov_type == 'diagonal':
            covariance_matrix = diagonal_covariance(train_data_set[:, :-1].T)

        for i in range(0, train_data_set.shape[0]-1):
            row = train_data_set[i, :-1]

            train_log_likelihoods[i, y] = gaussian_log_likelihood_calculation(prior, mean, covariance_matrix, row)

        for i in range(0, test_data_set.shape[0]-1):
            row = test_data_set[i, :-1]

            test_log_likelihoods[i, y] = gaussian_log_likelihood_calculation(prior, mean, covariance_matrix, row)


    train_prediction = log_likelihood_classify(train_log_likelihoods)
    test_prediction = log_likelihood_classify(test_log_likelihoods)

    training_accurate = numpy.sum(train_prediction == training_targets)
    training_accuracy = float(training_accurate) / float(training_targets.size)

    testing_accurate = numpy.sum(test_prediction == testing_targets)
    testing_accuracy = float(testing_accurate) / float(testing_targets.size)

    train_avg_likelihood = average_example_log_likelihood(train_log_likelihoods,
                                                          training_targets)
    test_avg_likelihood = average_example_log_likelihood(test_log_likelihoods,
                                                         testing_targets)

    fold_data = {'log L train': train_avg_likelihood,
                 'log L test': test_avg_likelihood,
                 'training accuracy': training_accuracy,
                 'testing accuracy': testing_accuracy}

    return fold_data

def gaussian_log_likelihood_calculation(prior, mean, covariance_matrix, features):

    inverse = numpy.linalg.inv(covariance_matrix)

    left = math.log(prior)

    dot = numpy.dot(features.T, inverse)
    mid = numpy.dot(dot, mean)

    dot = numpy.dot(mean.T, inverse)
    right = numpy.dot(dot, mean)

    return left + mid - 1/2 * right


def log_likelihood_classify(likelihood_array):

    prediction = numpy.zeros([likelihood_array.shape[0]], dtype=int)

    for i in range(0, likelihood_array.shape[0]):
        if likelihood_array[i, 0] > likelihood_array[i, 1]:
            prediction[i] = 0
        else:
            prediction[i] = 1

    return prediction


def diagonal_covariance(data_set):
    full_cov = numpy.cov(data_set)
    matrix = numpy.zeros([data_set.shape[0], data_set.shape[0]])

    for i in range(0, full_cov.shape[0]):
        matrix[i, i] = full_cov[i, i]

    return matrix


def kernelized_logistic_regression(training_features, training_targets, testing_features, testing_targets,
                                   precision=0.01, learning_rate=0.7, variance=1.0, max_iter=100):

    kernel = build_kernel_matrix(training_features, variance)

    alphas = numpy.zeros([training_features.shape[0], 1], dtype=float)

    hypothesis = klr_build_hypothesis_vector(training_features, alphas, variance)
    gradient = klr_gradient(kernel, training_targets, hypothesis)


    iteration = 0

    while precision < gradient.all() and iteration < max_iter:

        iteration += 1

        hypothesis = klr_build_hypothesis_vector(training_features, alphas, variance)
        gradient = klr_gradient(kernel, training_targets, hypothesis)

        alphas -= learning_rate * gradient


    # get the log likelihoods from the newly training alphas
    train_log_likelihoods = klr_build_log_likelihood_array(training_features, alphas, variance, training_features)
    test_log_likelihoods = klr_build_log_likelihood_array(training_features, alphas, variance, testing_features)

    # determine the predictions of the model
    train_prediction = log_likelihood_classify(train_log_likelihoods)
    test_prediction = log_likelihood_classify(test_log_likelihoods)

    # calculate the training accuracy of the model
    training_accurate = numpy.sum(train_prediction == training_targets)
    training_accuracy = float(training_accurate) / float(training_targets.size)

    # calculate the testing accuracy of the model
    testing_accurate = numpy.sum(test_prediction == testing_targets)
    testing_accuracy = float(testing_accurate) / float(testing_targets.size)

    train_avg_likelihood = average_example_log_likelihood(train_log_likelihoods,
                                                          training_targets)
    test_avg_likelihood = average_example_log_likelihood(test_log_likelihoods,
                                                         testing_targets)

    fold_data = {'log L train': train_avg_likelihood,
                 'log L test': test_avg_likelihood,
                 'training accuracy': training_accuracy,
                 'testing accuracy': testing_accuracy}

    return fold_data


def klr_build_log_likelihood_array(trained_features, alphas, variance, features):

    log_likelihoods = numpy.zeros([features.shape[0], 2], dtype=float)

    for i in range(0, features.shape[0]):
        row = features[i]

        log_likelihoods[i, 0] = klr_log_likelihood_calculation(trained_features, alphas, variance, row, 0)
        log_likelihoods[i, 1] = klr_log_likelihood_calculation(trained_features, alphas, variance, row, 1)

    return log_likelihoods


def klr_log_likelihood_calculation(features, alphas, variance, x_vector, y):
    if y == 1:
        return numpy.log(klr_hypothesis(features, alphas, variance, x_vector))
    elif y == 0:
        return numpy.log(1 - klr_hypothesis(features, alphas, variance, x_vector))

def klr_build_hypothesis_vector(training_features, alphas, variance):

    hypotheses = numpy.zeros([training_features.shape[0], 1], dtype='float')

    for i in range(0, training_features.shape[0]):
        row = training_features[i]
        hypotheses[i] = klr_hypothesis(training_features, alphas, variance, row)

    return hypotheses


def klr_hypothesis(features, alphas, variance, x_input):

    exponent = 0

    for i in range(0, features.shape[0]):
        exponent += alphas[i] * gaussian_kernel(features[i], x_input, variance)

    denominator = 1 + numpy.exp(exponent)

    return 1 / denominator


def build_kernel_matrix(features, variance):

    k = features.shape[0]

    kernel = numpy.zeros([k, k], dtype=float)

    for i in range(0, k):
        for j in range(0, k):
            kernel[i, j] = gaussian_kernel(features[i], features[j], variance)

    return kernel


def gaussian_kernel(vector_x, vector_z, variance):

    difference = (vector_x - vector_z)

    squared = numpy.linalg.norm(difference) ** 2

    division = -squared/(2*variance)

    return numpy.exp(division)


def klr_gradient(kernel, targets, hypotheses):

    error = targets[:, numpy.newaxis] - hypotheses

    gradient = numpy.dot(kernel.T, error)

    return gradient


def generate_latex_table(table_dict):
    print '\\begin{table}[h]'
    print ' \\begin{tabular}{l|c|c|c|c|c|}'
    print ' \\cline{2-6}'

    print '     & \\multicolumn{5}{c|}{Folds}      \\\\ \\cline{2-6}'
    print '     &  1  &  2  &  3  &  4  &  5 \\\\ \hline'

    print '\\multicolumn{1}{|c|}{log L Train} & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f  \\\\ \\hline' % \
          (table_dict[0]['log L train'],
           table_dict[1]['log L train'],
           table_dict[2]['log L train'],
           table_dict[3]['log L train'],
           table_dict[4]['log L train'])

    print '\\multicolumn{1}{|c|}{log L Test} & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f  \\\\ \\hline' % \
          (table_dict[0]['log L test'],
           table_dict[1]['log L test'],
           table_dict[2]['log L test'],
           table_dict[3]['log L test'],
           table_dict[4]['log L test'])


    print '\\multicolumn{1}{|l|}{Training Accuracy} & %4.2f\\%% & %4.2f\\%% & %4.2f\\%% & %4.2f\\%% & %4.2f\\%% \\\\ \\hline' % \
          (table_dict[0]['training accuracy'],
           table_dict[1]['training accuracy'],
           table_dict[2]['training accuracy'],
           table_dict[3]['training accuracy'],
           table_dict[4]['training accuracy'])

    print '\\multicolumn{1}{|l|}{Testing Accuracy} & %4.2f\\%% & %4.2f\\%% & %4.2f\\%% & %4.2f\\%% & %4.2f\\%% \\\\ \\hline' % \
          (table_dict[0]['testing accuracy'],
           table_dict[1]['testing accuracy'],
           table_dict[2]['testing accuracy'],
           table_dict[3]['testing accuracy'],
           table_dict[4]['testing accuracy'])

    print '\\end{tabular}'
    print '\\end{table}'


if __name__ == '__main__':

    array_x = numpy.loadtxt('wpbcx.dat', float)
    vector_y = numpy.loadtxt('wpbcy.dat', float)

    #############
    # question 4 a
    #############
    stats = five_fold_cross_validation(array_x, vector_y, model='logistic')
    generate_latex_table(stats)

    #############
    # question 4 b
    #############
    stats = five_fold_cross_validation(array_x, vector_y, model='gaussian', cov_type='diagonal')
    generate_latex_table(stats)

    #############
    # question 4 c
    #############
    stats = five_fold_cross_validation(array_x, vector_y, model='gaussian')
    generate_latex_table(stats)

    #############
    # question 4 d
    #############
    stats = five_fold_cross_validation(array_x, vector_y, model='klr')
    generate_latex_table(stats)
