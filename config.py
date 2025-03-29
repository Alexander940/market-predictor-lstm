"""
Archivo de configuración para el proyecto de LSTM de predicción de mercado.
"""

import os
import torch

# Rutas
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
CODE_DIR = os.path.join(os.path.dirname(__file__), 'code')

# Datos
DATA_FILE = os.path.join(DATA_DIR, 'two-years-data.csv')
DATA_SEPARATOR = ';'
FEATURES = ['close']
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M'

# Preprocesamiento
TIME_STEPS = 15  # Secuencia para predicción
INTERVAL_MINUTES = 10  # Agrupación de datos en intervalos

# Modelo
HIDDEN_SIZE = 64
NUM_LAYERS = 3
DROPOUT_RATE = 0.3
BIDIRECTIONAL = True

# Entrenamiento
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
PATIENCE = 10
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Predicción
PREDICTION_HORIZON = 20  # Número de intervalos a predecir

# Dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)