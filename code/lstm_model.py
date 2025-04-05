import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def predict(model, x, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Asegurar que x tenga la forma correcta (batch_size, seq_len, input_dim)
        if isinstance(x, np.ndarray):
            if len(x.shape) == 2:
                x = x.reshape(1, x.shape[0], x.shape[1])
            x = torch.FloatTensor(x)
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 2:
                x = x.reshape(1, x.shape[0], x.shape[1])
        
        x = x.to(device)
        predictions = model(x)
        return predictions.cpu().numpy() 