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


dsFactory = FileDataSourceFactory(filename="../../../../data/textualMNIST/mnist_train.txt", decoder=MNISTDecoder(),
                                      numberOfNodes=1, indices='roundRobin', shuffle=False, cache=False)
dataSource=dsFactory.getDataSource(0)
dataSource.prepare()
data=dataSource.getNext()
print(data)


model_new = DropoutNet()
model_new.load_state_dict(torch.load('weights_only.pth'))
#image = torch.from_numpy(image)
exampleTensor = torch.FloatTensor(data[0])
exampleTensor.unsqueeze(3)
model_new.predict(exampleTensor)
print(model_new)
