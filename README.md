# Modelo LSTM para Predicción de Mercado

Este proyecto implementa un modelo LSTM (Long Short-Term Memory) para predecir movimientos de mercado basados en datos históricos.

## Estructura del Proyecto

```
LSTM/
├── code/                    # Código fuente y notebooks
│   ├── market-prediction-pytorch.ipynb  # Notebook principal
│   ├── model_training.py    # Script de entrenamiento
│   ├── prediction.py        # Script de predicción
│   └── utils.py             # Funciones de utilidad
├── data/                    # Datos de mercado
│   └── two-years-data.csv   # Datos históricos (2 años)
├── models/                  # Modelos guardados
├── config.py                # Configuración del proyecto
└── requirements.txt         # Dependencias
```

## Instalación

1. Clone este repositorio
2. Instale las dependencias:
```
pip install -r requirements.txt
```

## Uso

### Entrenamiento del modelo

```bash
python code/model_training.py
```

### Realizar predicciones

```bash
python code/prediction.py
```

### Notebook

Para explorar el modelo de forma interactiva, abra el notebook:

```bash
jupyter notebook code/market-prediction-pytorch.ipynb
```

## Características del Modelo

- LSTM bidireccional con 3 capas
- Normalización de capa y batch
- Regularización con dropout (0.3-0.4)
- Optimizador AdamW con reducción dinámica de learning rate
- Early stopping para evitar sobreajuste
