# Predicción de Churn en Telco — Examen Final, Análisis Predictivo (ITBA)

Modelo predictivo de fuga de clientes (*churn*) sobre el dataset público
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7.043 clientes × 21 variables, target `Churn`).

**Caso de negocio:** identificar qué clientes tienen alta probabilidad de darse de baja el
próximo mes para que el equipo de retención actúe proactivamente. El modelo produce un
score mensual por cliente; el umbral de acción es una decisión de negocio (el score permite
priorizar según la capacidad operativa del equipo).

## Resultado

**Modelo final: LightGBM** (features base, hiperparámetros regularizados vía búsqueda
aleatoria). Evaluación única sobre test (20%, estratificado, nunca usado en el desarrollo):

| Métrica | Valor |
|---|---|
| PR-AUC | 0,666 (baseline azar: 0,265) |
| Recall (umbral operativo 0,153, elegido por F2) | 89,6% |
| Precision | 45,3% |
| F2 | 0,749 |

## Estructura

| Archivo | Contenido |
|---|---|
| `01_eda.ipynb` | Análisis exploratorio: limpieza, balance de clases, drivers de churn |
| `02_experimentos.ipynb` | Selección de modelos: 10 algoritmos × 2 configs de features (defaults), tuning de los 5 mejores, ensambles, CatBoost nativo, FE v2, elección de umbral por F2 |
| `03_entrenar_modelo_final.ipynb` | **Entrega b:** entrena y guarda `modelo_final.joblib` + `umbral_operativo.json` (reproducible al byte) |
| `04_aplicar_modelo.ipynb` | **Entrega c:** aplica el modelo guardado al test y genera `predicciones_test.csv` (replicable exactamente) |
| `05_evaluacion_final.ipynb` | Única evaluación en test + curva de ganancia + calibración + interpretación SHAP |

## Reproducibilidad

```
pip install -r requirements.txt
```

Ejecutar los notebooks en orden (`01` → `05`). Todo usa semilla 42 y partición
estratificada 80/20 fija; re-ejecutar `03` y `04` regenera el modelo y las predicciones
**byte a byte** (verificado por hash SHA-256).

- Métrica de comparación: **PR-AUC** (average precision), apropiada para clases
  desbalanceadas (26,5% churn). Umbral operativo elegido por **F2** (prioriza recall:
  perder un cliente sin intentar retenerlo cuesta más que una oferta de más).
- Validación: CV estratificada de 5 folds dentro de train para todas las decisiones;
  el test se usa una única vez en `05`.
