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

'''
    SVM in scikit:
        SVC
        NuSVC
        LinearSVC: does not accept kernel
        
        input:
        ------
        X = [n_samples, n_features]  -> this holds the training samples
        Y = [n_samples]              -> this holds the class label
        
        svm.properties:
        ---------------
        support_vectors_    //get support vector
        support_            //indices of support vector
        n_support_          //number of support vector
        
        df_shape:
        --------
        SVC & NuSVC:
            one against one approach in multi class classification :ovo
            svm.SVC().decision_function([1]).shape[1] = n_class * (nclass-1)/2
        Linear SVC:
            one against the rest: ovr
                svm.LinearSVC().decision_function([1]).shape([1])
                
        unbalance_problems:
        ---------------------
            implement a keyword: class_weight => {class_label: value(float)}
         
        Kernel function:
        -------------------
            Linear:<X,X'>
            Polynomial:< r <X,X'> + r >^d, d is degree, r is coef0
            rbf: exp(-r||X-X'||^2), r is gamma, r > 0
            sigmoid: tanh (r<X,X"> + R), R is coef0
            customize kernel function:
                >>> def my_kernel(X, Y):
                ...     return np.dot(X, Y.T)
                ...
                >>> clf = svm.SVC(kernel=my_kernel)
        
                     

'''
