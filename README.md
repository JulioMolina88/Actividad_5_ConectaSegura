# Actividad 5: predicción de abandono de clientes — ConectaSegura

Proyecto académico de clasificación para estimar el abandono de clientes de una empresa de telecomunicaciones. Esta actividad usa abandono de clientes (churn), no detección de llamadas fraudulentas.

## Contenido

- `fuentes/entrena.ipynb`: cuaderno ejecutado en Colab, con sus salidas originales.
- `fuentes/datos_prep.py`: descarga y preparación básica de datos.
- `fuentes/train.py`: entrenamiento de regresión logística y bosque aleatorio, búsqueda de hiperparámetros y registro en MLflow.
- `evidencias/`: capturas de la ejecución original y comparación estadística.
- `informes/`: reporte técnico en PDF.
- `datos/`: carpetas para datos originales y limpios, que se generan al reproducir el proyecto.

## Datos y decisiones

Fuente: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data). La variable objetivo es `Churn` (Yes/No). Se excluye `customerID`; `TotalCharges` se convierte a numérico y los valores vacíos se tratan como nulos. La imputación, el escalado numérico y la codificación categórica se ajustan dentro de cada partición de entrenamiento para evitar fugas. El script de descarga consulta la versión disponible en la fuente; el hash SHA-256 del archivo usado queda registrado en MLflow.

## Reproducción local

Requiere Python, conexión a Internet para instalar dependencias y descargar el CSV. Desde la carpeta del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fuentes/datos_prep.py
python fuentes/train.py
```

El entrenamiento usa 80 % para entrenamiento y 20 % para prueba, partición estratificada con semilla 42, y búsqueda con validación cruzada de tres pliegues para maximizar F1. La cuadrícula completa del bosque puede tardar. Para comprobar que el flujo funciona antes de ejecutarla se puede usar `python fuentes/train.py --rapido`; este modo no reproduce la búsqueda completa.

El script imprime el comando exacto para abrir la interfaz de MLflow. Desde la raíz del proyecto, normalmente es:

```bash
mlflow ui --backend-store-uri sqlite:///resultados/mlflow.db
```

Abrir la dirección local que muestre MLflow y seleccionar `ConectaSegura_Abandono_Clientes`. Se registran parámetros, métricas, modelo, tabla de validación y hash del archivo. La carpeta `resultados/` es local y está excluida de Git; las capturas incluidas documentan la ejecución anterior. Los archivos de datos tampoco se suben: el script permite obtenerlos nuevamente.

## Cuaderno de Colab

El cuaderno incluido conserva la ejecución original y usa rutas de Colab como `/content/Actividad5`. Si se reinicia la sesión, restaura los archivos y revisa sus rutas y dependencias antes de ejecutar. Los scripts locales anteriores son una alternativa para reproducir el flujo; sus valores pueden variar con las versiones instaladas o cambios en los datos. La comparación bootstrap mostrada en las evidencias procede de la ejecución original, no de `train.py`.

## Resultados de la ejecución original

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Bosque aleatorio | 0.7672 | 0.5458 | 0.7326 | 0.6256 | 0.8406 |

La diferencia observada de F1 (bosque menos logística) tuvo un intervalo bootstrap reportado de aproximadamente [-0.0112, 0.0346], que incluye cero. Estos valores no garantizan resultados idénticos al ejecutar nuevamente.
