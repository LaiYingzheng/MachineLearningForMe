'''
    kNN(k-NerestNeighbor):

            Collect k nearest samples from training sets, find the most frequent type

    --------

    Target:
            Catagorize unknow samples.

    Input:
            uncategorized samples,
            sample set D,
            known sample j.

    Ouput:
            Catagorized sample.

    Pro:
            Easy implementation

    Con:
            Tons of computation

    Note:
            K <= 20
'''



from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, Labels, k):
    '''
        inX     -- testing vector
        dataSet -- training set, one row for one sample
        Label   -- correspounding label vectors for dataset
        k       -- the closest neighbour number
    '''
    dataSetSize = dataSet.shape[0]                       # Get dataset row number, sample number
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet      # tile(A,(m,n)), use array A to construct m * n matrix
    sqDiffMat = diffMat **2
    sqDistances = sqDiffMat.sum(axis = 1)                # array.sum(axis = 1), accumulating number through row, Axis = 0, through coloums
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance.argsort()              # Get order number for every element

    classCount={}                                        # sortedDistIndicies[0], this shows the order number of the first number in the number array

    for i in range(k):
        voteIlabel = Labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get(key,x), get key from correspounding value x, otherwise return 0
    sortedClassCounts = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCounts[0][0]


# Transfer image txt file into vector
def img2vector(filename):
    Vector = zeros((1,1024))
    file = open(filename)
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            Vector[0,32*i+j] = int(lineStr[j])

    return Vector

#Load Traning set into big vector

def handwritingClassTest():
    hwLabels = []
    trainFileList = listdir('trainingDigits')
    m = len(trainFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' %fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mT = len(testFileList)
    for i in range(mT):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        VectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classifierResult = classify0(VectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("the total number of error is : %d" % errorCount)
    print("the total error rate is: %d" % (errorCount/float(mT)))