__author__ = 'geoffrey'

import unittest
import numpy
import assignment2

class TestAssignment1(unittest.TestCase):

    def test_gaussian_multivariate_exponent_calculation(self):

        row = numpy.array([[1], [2]])
        row_mean = numpy.array([[2], [2]])
        covariance_matrix = numpy.array([[11.71, -4.286], [-4.286, 2.144]])

        result = assignment2.gaussian_multivariate_exponent_calculation(row, row_mean, covariance_matrix)


        expected = 1.1724955240019641

        self.assertEquals(result[0, 0], expected)


if __name__ == '__main__':

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAssignment1)
    unittest.TextTestRunner(verbosity=2).run(suite)