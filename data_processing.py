#librerias
import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np


#abrir archivo

dividendos = pd.read_csv('Data/raw.csv')
#limpieza archivo


df = dividendos.drop(columns=['Stock code'])

df['P/E ratio'] = pd.to_numeric(df['P/E ratio'], errors='coerce')

df = df[df['P/E ratio'].notna()]

processed = df[['Share pledge ratio of controlling shareholders',
                    'Pledge ratio of unlimited shares',
                    'audit opinion ',
                    'Downgrade or negative',
                    'Ratio of other receivables to total assets',
                    'ROA',
                    'Asset liability ratio (total liabilities - contract liabilities - advance receipts)/(total assets - goodwill - contract liabilities - advance receipts)',
                    'Pledge ratio of limited sale shares',
                    'ROE',
                    'Enterprise age',
                    'IsDefault']]

processed.rename(columns={"Asset liability ratio (total liabilities - contract liabilities - advance receipts)/(total assets - goodwill - contract liabilities - advance receipts)": "Asset liability ratio"}, inplace=True)

#guardar csv con datos limpios

processed.to_csv('processed.csv', index=False, encoding='utf-8')