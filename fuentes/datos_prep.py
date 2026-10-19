"""Descarga y limpieza del conjunto IBM Telco Customer Churn."""
from pathlib import Path
import argparse
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = ('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/'
       'master/data/Telco-Customer-Churn.csv')

def preparar(origen, destino):
    df = pd.read_csv(origen)
    if 'Churn' not in df or 'TotalCharges' not in df:
        raise ValueError('El archivo no contiene las columnas esperadas.')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    print(f'Datos preparados: {df.shape[0]} filas, {df.shape[1]} columnas; '
          f'TotalCharges nulos: {df.TotalCharges.isna().sum()}. Guardado en {destino}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entrada', type=Path, default=ROOT/'datos/datos_ini/Telco-Customer-Churn.csv')
    parser.add_argument('--salida', type=Path, default=ROOT/'datos/datos_limp/telco_limpio.csv')
    args = parser.parse_args()
    if not args.entrada.exists():
        args.entrada.parent.mkdir(parents=True, exist_ok=True)
        print(f'Descargando datos desde {URL}')
        urllib.request.urlretrieve(URL, args.entrada)
    preparar(args.entrada, args.salida)

if __name__ == '__main__':
    main()
