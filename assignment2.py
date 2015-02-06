__author__ = 'geoffrey'

import numpy
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

if __name__ == '__main__':

    array_x = numpy.loadtxt('wpbcx.dat', float)
    vector_y = numpy.loadtxt('wpbcy.dat', float)

    five_fold_cross_validation(array_x, vector_y)

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