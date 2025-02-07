import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch_tools import Module

class KMNISTModel(Module):
    
    def __init__(self):
    
        super(KMNISTModel, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear( 28 * 28, 128 )  # KMNIST images are 28x28
        self.linear2 = nn.Linear( 128, 64 )
        self.linear3 = nn.Linear( 64, 10 )  # 10 classes for KMNIST
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax( dim = 1 )
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam( self.parameters(), lr = 0.001 )

    def forward(self, x):

        x = self.flatten( x )
        x = self.relu( self.linear1( x ) )
        x = self.relu( self.linear2( x ) )
        x = self.softmax( self.linear3( x ) )

        return x
    
    def forward_backward(self, inputs, labels):
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self( inputs )
        
        # Calculate loss
        loss = self.criterion( outputs, labels )
                
        # Backward pass
        loss.backward()
        
        
        return loss.item()
