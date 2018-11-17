
'''
input: - NAME_features.npy contains a numpy array of feature data with
        'n' rows and 'm' columns
       - NAME_labels.npy contains a numpy array of label data with 
        'n' rows and 'q' columns

careful with large; we're using separate neural networks for each label, which 
may not capture the dependencies of the labels

output: - NAME_modelWeights.hd5 contains the weights of the neural network
        - NAME_results.txt contains an overview of the resulting training
            and testing results of the neural network

'''

## imports
import argparse
import numpy as np

# importing keras
print("Importing neural network libraries")
import keras
from keras.models import Sequential
from keras.layers import Dense



'''
input: dimension of a single feature vector
output: model dictated by the topology of the neural network

highly variable, should be tweaked to obtain the optimal neural network
should be pretty obvious what this function does
eg. add more layers
eg. change activation
eg. add MOAR layers
'''
def setupModel(inputDim):
    n = 5
    model = Sequential()
    model.add(Dense(n, input_dim=inputDim, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(n, activation='tanh'))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

'''
input: trainData - the features that the models are trained on
        trainLabels - labels that the features should match
        model - if the user wishes to use a preexisting model structure
                optional
output:
        model after it has finished training

the function trains the model on the data, again, pretty simple
'''
def trainModel(trainData, trainLabels, model=0):
    if(model==0):
        model = setupModel(trainData.shape[1])
    model.fit(trainData, trainLabels, epochs = 1000, batch_size=32)
    
    return model

'''
input:
        testData - features that we use to feed into the model
        testLabels - what the features should match
        model - model we're testing, should've been made in trainModel
output:
        whatever loss_and_metrics is supposed to be

tests the model, we can probably do more fancy things like cross validation
and whatnot eventually, but we'll keep it simple for now
'''
def testModel(testData, testLabels, model):
    loss_and_metrics = model.evaluate(testData, testLabels)

    ## eventually use this to determine more complex matrix or look at specific instances
    classes = model.predict(testData, batch_size=128)
    print(classes)
    print(testLabels)
    print(loss_and_metrics)
    return loss_and_metrics

'''
input:
        featureData - 'n' by 'm' matrix with 'n' rows of features, each of which
                        is 'm' diminesional
        featureLabels - 'n' by 'q' matrix with 'n' rows of labels, each of which is
                        'q' dimensional
        trainPercentage - how much of the data the user wishes to use for training
                        versus testing
output: 
        trainData - n*% by m
        trainLabels - n*% by q
        testData - n*(1-%) by m
        testLables - n*(1-%) by q
'''

def splitTestData(featureData, featureLabels, trainPercentage):
    ## divide up the traindata
    numInstances, numLabels = featureLabels.shape
    numInstances2, numFeatures = featureData.shape

    # check that each feature vector corresopnds to a label vector
    if(not numInstances == numInstances2):
        print('label has ' + str(numInstances2) + ' instances while ' + 
                'feature has ' + str(numInstances) + ' instances')
        return
    
    ## TODO: evnentually randomize this
    trainIndices = np.arange(int(trainPercentage*numInstances))
    testIndices = np.setxor1d(trainIndices, np.arange(numInstances))
    
    # split up data
    trainData = featureData[trainIndices,:]
    trainLabels = featureLabels[trainIndices,:]
    testData = featureData[testIndices,:]
    testLabels = featureLabels[testIndices,:]
    
    return trainData, trainLabels, testData, testLabels


'''
main function reads in the arguments and whatnot
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('featureData', help='path to feature data')
    parser.add_argument('featureLabels', help='path to feature labels')
    args = parser.parse_args()
    
    featureData = np.load(args.featureData)
    featureLabels = np.load(args.featureLabels)
 
    # divide data into train and test
    trainPercentage = 0.8
    trainData, trainLabels, testData, testLabels = splitTestData(
                featureData, featureLabels, trainPercentage)

    #train and test
    neuralNet = trainModel(trainData, trainLabels[:,0])
    loss_metric = testModel(featureData, featureLabels[:,0], neuralNet)


if __name__=='__main__':
    main()
