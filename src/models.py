import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(32, 32), activation="ReLU", dropout=0.0):
        super().__init__()
        act = getattr(nn, activation)
        camadas = []
        d = in_dim
        
        for h in hidden:
            camadas.extend([nn.Linear(d, h), act()])
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            d = h
            
        camadas.append(nn.Linear(d, out_dim)) # Saída em logits (sem ativação final aqui)
        self.net = nn.Sequential(*camadas)

    def forward(self, x):
        return self.net(x)