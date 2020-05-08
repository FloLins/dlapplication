import sys
sys.path.append("../../../../../dlapplication")
sys.path.append("../../../../../dlplatform")
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
from environments.local_environment import Experiment
from environments.datasources import FileDataSourceFactory
from dlutils.models.pytorch.dropOutNetwork import DropoutNet
from DLplatform.synchronizing import PeriodicSync
from DLplatform.aggregating import Average
from DLplatform.learning.factories.pytorchLearnerFactory import PytorchLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler
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






dsFactory = FileDataSourceFactory(filename="../../../../data/textualMNIST/mnist_train.txt", decoder=MNISTDecoder(),
                                  numberOfNodes=1, indices='roundRobin', shuffle=False, cache=False)
dataSource=dsFactory.getDataSource(0)
dataSource.prepare()


model_new = DropoutNet()
model_new.load_state_dict(torch.load('weights_only.pth'))

for i_ in range(1):
    exampleTensor, label = get_next_sample(dataSource)
    prediction, prediction_probs = model_new.predict(exampleTensor)
    print("Predicted: " + str(prediction) + " for label: " + str(label))
    variance_for_every_classifier=functions.calculate_variance(prediction_probs)
    deviation_for_every_classifier=functions.calculate_deviation(variance_for_every_classifier)



print(model_new)

