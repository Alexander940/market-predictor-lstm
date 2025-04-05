import torch
import numpy as np
from lstm_model import LSTMModel, predict

# Configuración del modelo
input_dim = 13  # Dimensión de entrada
hidden_dim = 128  # Dimensión oculta
output_dim = 6  # Dimensión de salida
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Crear modelo
model = LSTMModel(input_dim, hidden_dim, output_dim)
model = model.to(device)

# Crear datos de prueba
x = np.random.rand(100, 13)  # 100 muestras, 13 características

try:
    # Intentar predecir con los datos
    predictions = predict(model, x, device)
    print("Forma de la entrada:", x.shape)
    print("Forma de la salida:", predictions.shape)
    print("¡La predicción fue exitosa!")
except Exception as e:
    print("Error durante la predicción:", str(e)) 