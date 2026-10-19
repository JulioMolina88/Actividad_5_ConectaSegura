"""Entrena modelos de abandono y registra experimentos locales en MLflow."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datos', type=Path, default=ROOT/'datos/datos_limp/telco_limpio.csv')
    parser.add_argument('--salida', type=Path, default=ROOT/'resultados')
    parser.add_argument('--rapido', action='store_true', help='Prueba con una sola combinación por modelo')
    args = parser.parse_args()
    df = pd.read_csv(args.datos)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    if y.isna().any():
        raise ValueError('Churn debe contener exclusivamente Yes y No.')
    X = df.drop(columns=['Churn', 'customerID'], errors='ignore').copy()
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    numericas = X.select_dtypes(include='number').columns.tolist()
    categoricas = X.select_dtypes(exclude='number').columns.tolist()
    pre = ColumnTransformer([
        ('num', Pipeline([('imputar', SimpleImputer(strategy='median')), ('escalar', StandardScaler())]), numericas),
        ('cat', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categoricas)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    modelos = [
        ('Regresion_logistica', LogisticRegression(max_iter=2000, random_state=42),
         {'modelo__C': [.1, 1, 10], 'modelo__class_weight': [None, 'balanced']}),
        ('Bosque_aleatorio', RandomForestClassifier(random_state=42, n_jobs=1),
         {'modelo__n_estimators': [100, 200], 'modelo__max_depth': [None, 10],
          'modelo__min_samples_leaf': [1, 3], 'modelo__class_weight': [None, 'balanced']})]
    args.salida.mkdir(parents=True, exist_ok=True)
    tracking = (args.salida/'mlflow.db').resolve()
    mlflow.set_tracking_uri('sqlite:///' + tracking.as_posix())
    nombre = 'ConectaSegura_Abandono_Clientes'
    experimento = mlflow.get_experiment_by_name(nombre)
    if experimento is None:
        mlflow.create_experiment(nombre, artifact_location=(args.salida/'artefactos').resolve().as_uri())
    mlflow.set_experiment(nombre)
    resumen = []
    for nombre_modelo, modelo, grid in modelos:
        if args.rapido:
            grid = {k: [v[0]] for k, v in grid.items()}
        pipe = Pipeline([('preprocesamiento', pre), ('modelo', modelo)])
        busqueda = GridSearchCV(pipe, grid, cv=cv, scoring='f1', n_jobs=1, refit=True)
        busqueda.fit(Xtr, ytr)
        pred = busqueda.predict(Xte)
        prob = busqueda.predict_proba(Xte)[:, 1]
        metricas = {'f1_cv': busqueda.best_score_, 'accuracy': accuracy_score(yte, pred),
                    'precision': precision_score(yte, pred, zero_division=0),
                    'recall': recall_score(yte, pred, zero_division=0),
                    'f1': f1_score(yte, pred, zero_division=0), 'roc_auc': roc_auc_score(yte, prob)}
        ruta_cv = args.salida/f'{nombre_modelo}_cv.csv'
        pd.DataFrame(busqueda.cv_results_).to_csv(ruta_cv, index=False)
        with mlflow.start_run(run_name=nombre_modelo):
            mlflow.log_params({k: str(v) for k, v in busqueda.best_params_.items()})
            mlflow.log_param('sha256_datos', hashlib.sha256(args.datos.read_bytes()).hexdigest())
            mlflow.log_param('modo', 'rapido' if args.rapido else 'completo')
            mlflow.log_metrics(metricas)
            mlflow.log_artifact(str(ruta_cv))
            mlflow.sklearn.log_model(busqueda.best_estimator_, 'modelo')
        resumen.append({'modelo': nombre_modelo, **metricas})
        print(nombre_modelo, json.dumps(metricas, ensure_ascii=False))
    pd.DataFrame(resumen).to_csv(args.salida/'comparacion.csv', index=False)
    print(f'Resultados en {args.salida.resolve()}')
    print(f'Para abrir MLflow: mlflow ui --backend-store-uri sqlite:///{tracking.as_posix()}')

if __name__ == '__main__':
    main()
