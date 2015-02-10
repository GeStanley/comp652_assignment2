__author__ = 'geoffrey'

import unittest
import numpy
import assignment2

class TestAssignment1(unittest.TestCase):

    def test_covariance(self):

        dataset1 = numpy.array([[1,2,3,4,5,6,7,8,9,10]])
        dataset2 = numpy.array([[10,9,8,7,6,5,4,3,2,1]])


        expected = numpy.cov(dataset1, dataset2)


        result = assignment2.covariance(dataset1, dataset2)

        print expected
        print result

    # def test_build_normalized_array(self):
    #
    #     dataset = numpy.array([[2, -10, 1],
    #                             [4, 6, 1],
    #                             [10, 8, 1],
    #                             [-8, 7, 1],
    #                             [8, -5, 1]], dtype=float)
    #
    #     result = load_data.BuildNormalizedArray(dataset)
    #
    #     expected_result = numpy.array([[0.2, -1, 1],
    #                             [0.4, 0.6, 1],
    #                             [1, 0.8, 1],
    #                             [-0.8, 0.7, 1],
    #                             [0.8, -0.5, 1]])
    #
    #     self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-01, atol=1e-01))



suite = unittest.TestLoader().loadTestsFromTestCase(TestAssignment1)
unittest.TextTestRunner(verbosity=2).run(suite)