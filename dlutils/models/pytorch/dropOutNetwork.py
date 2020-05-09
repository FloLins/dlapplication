import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random


class DropoutNet(nn.Module):
    def __init__(self):
        super(DropoutNet, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        init.xavier_normal_(self.conv1.weight.data)
        init.zeros_(self.conv1.bias.data)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        init.xavier_normal_(self.conv2.weight.data)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        init.xavier_normal_(self.fc1.weight.data)
        self.fc2 = nn.Linear(50, 10)
        init.xavier_normal_(self.fc2.weight.data)


    def forward(self, x):
        h1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        h1 = F.dropout(h1, p=0.5, training=self.training)

        h2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(h1)), 2))
        h2 = F.dropout(h2, p=0.5, training=self.training)

        h3 = h2.view(-1, 320)
        h3 = F.relu(self.fc1(h3))
        h3 = F.dropout(h3, p=0.5, training=self.training)

        h4 = self.fc2(h3)
        h4 = F.dropout(h4, p=0.5, training=self.training)
        self.safe_model()
        return F.log_softmax(h4, dim=1)

    def __str__(self):
        return "MNIST simple Dropout Regularization inheritated from MNISTNetwork"

    def safe_model(self):
        #print("Safe the model")
        #if(self.training==False):
            #print("Safe the model")
        torch.save(self.state_dict(), './weights_only.pth')
        #torch.save(self, 'entire_model.pth')

    def predict(self, X, times=3):
        self.train(True)
        summed_probs=[0] * 10
        predictions_prob=[]
        for _ in range(times):
            output = self.forward(X)
            probs = torch.exp(output)
            array = probs[0].data.numpy()
            predictions_prob.append(array)
            summed_probs=summed_probs + array

        prediction = (np.where(summed_probs == np.amax(summed_probs)))
        prediction = prediction[0][0]
        #print(prediction)
        self.train(False)
        return prediction, predictions_prob, summed_probs




