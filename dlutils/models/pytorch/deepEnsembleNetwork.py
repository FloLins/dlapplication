import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random

class MnistNet(nn.Module):
    def __init__(self):
        self.seed=np.random.randint(0,100)
        #self.seed=42
        super(MnistNet, self).__init__()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic=True

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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #self.safe_model()
        return F.log_softmax(x, dim=1)

    def __str__(self):
        return "MNIST simple CNN with Safe for Deep Ensemble"

    def safe_model(self):
        PATH = './'+str(42)+'_weights_only.pth'
        torch.save(self.state_dict(), PATH)
