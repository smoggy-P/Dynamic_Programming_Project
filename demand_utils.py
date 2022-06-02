import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, 
            size, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.size = size
        self.n_layers = n_layers
        self.output_activation = output_activation
        
        layers_size = [self.input_size] + ([self.size]*self.n_layers) + [self.output_size]
        self.layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i+1]) 
                                    for i in range(len(layers_size)-1)])
        
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            if i!=len(self.layers)-1:
                out = self.activation(layer(out))
            else:
                out = layer(out)

        if self.output_activation is not None:
            out = self.output_activation(out)
       
        return out

class Demand_Model(nn.Module):
    def __init__(self, device):
        super(Demand_Model, self).__init__()
        self.device = device
        self.learning_rate = 0.01
        mean = MLP(input_size=3, output_size=1, n_layers=2, size=128).to(self.device)
        logstd = torch.zeros(1, requires_grad=True, device=self.device)
        self.parameters = (mean, logstd)
        self.optimizer = optim.Adam([logstd] + list(mean.parameters()), lr=self.learning_rate)
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')

    def transition_step(self, features):
        mean, logstd = self.parameters
        probs_out = mean(features)
        sample_out = probs_out + torch.exp(logstd) * torch.randn(probs_out.size(), device=self.device)
        return sample_out
    
    def update(self, features, labels):
        sample_out = self.transition_step(features)
        loss = self.mse_criterion(sample_out, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class Demand_Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
