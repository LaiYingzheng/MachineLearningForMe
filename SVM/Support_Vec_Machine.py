'''
    SVM:
            Supervised Machine Learning Algorighm: Classfication problem

    Classification:
            Finding the Hyperplane that differentiates the two classes.

            Optimal Hyper-plane:
                    Maximize the margins from both tags,
                    The distance from the hyperplane to nearest element of each tag is the largest.

            Linear -> 2D hyper-plane
            Non-Linear -> 3D hyper-plane

    Parameters:
            Kernel:
                Kernel trick:
                    polynomial and exponential kernels calculates separation
                    line in higher dimension

            Regularization
                For Large values of this parameter,
                    optimization will choose a smaller-margin hyperplane to classfied all the data correctly
                For small values,
                    optimizer will lookfor a larger-margin hyper plane, even if it is misclassified more points

            Gamma
                This defines how far the influence of a single training set reaches.
                    low Gamma:
                        points far away from the possible separation line
                        are considered in calculation for the separation line
                    High Gamma:
                        the points close to possible separation line are considered into calculation

            Margin
                A margin is a separation of line to the closet class points
                A good margin is one where this separation is larger for both the classes



'''
from libsvm.svmutil import *

