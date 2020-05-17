import sys
sys.path.append("../../../../../dlapplication")
sys.path.append("../../../../../dlplatform")
import torch
import numpy as np
from environments.datasources import FileDataSourceFactory
from dlutils.models.pytorch.deepEnsembleNetwork import MnistNet
from dlapplication.environments.datasources.dataDecoders.pytorchDataDecoders import MNISTDecoder
import functions


def get_next_sample(dataSource):
    data = dataSource.getNext()
    image = data[0]
    image = image[np.newaxis, ...]
    label = data[1]
    # print(image.shape)
    exampleTensor = torch.FloatTensor(image)
    return exampleTensor, label


def prepareDataSource():
    dsFactory = FileDataSourceFactory(filename="../../../../data/textualMNIST/mnist_train.txt", decoder=MNISTDecoder(),
                                  numberOfNodes=1, indices='roundRobin', shuffle=False, cache=False)
    dataSource=dsFactory.getDataSource(0)
    dataSource.prepare()
    return dataSource


def loadModels(countModels):
    models=[]
    for i in range(countModels):
        model = MnistNet()
        path = str(i) + 'model.pth'
        model.load_state_dict(torch.load(path))
        models.append(model)
    return models


def predict(models, x):
    summed_probs = [0] * 10
    predictions_prob = []
    for i in range(len(models)):
        output = models[i].forward(x)
        #print(output)
        probs = torch.exp(output)
        array = probs[0].data.numpy()
        predictions_prob.append(array)
        summed_probs = summed_probs + array

    prediction = (np.where(summed_probs == np.amax(summed_probs)))
    prediction = prediction[0][0]
    # print(prediction)
    return prediction, predictions_prob, summed_probs


dataSource = prepareDataSource()
models = loadModels(2)

for i_ in range(20):
    exampleTensor, label = get_next_sample(dataSource)
    prediction, prediction_probs, summed_probs = predict(models, exampleTensor)
    print("Predicted: " + str(prediction) + " for label: " + str(label))
    variance_for_every_classifier=functions.calculate_variance(prediction_probs)
    deviation_for_every_classifier=functions.calculate_deviation(variance_for_every_classifier)
    shannon_entropy=functions.compute_shannon_entropie(summed_probs)
    min_entropy = functions.compute_min_entropy(summed_probs)
    guessing_entropy=functions.compute_guessing_entropy(summed_probs)


