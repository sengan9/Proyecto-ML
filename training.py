#librerias
import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split

#abrir archivo

dividendos = pd.read_csv('Processed\processed.csv')

#guardar archivos train y test

train_data, test_data = train_test_split(processed, test_size=0.2, random_state=42, stratify=processed['IsDefault'])

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)