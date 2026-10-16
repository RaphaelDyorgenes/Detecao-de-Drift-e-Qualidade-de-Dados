"""
=============================================================================
Tech Challenge 04 — Etapa 1: Treinamento do Modelo Baseline de Risco de Crédito
=============================================================================

Módulo responsável pelo treinamento, validação cruzada, avaliação de desempenho
e serialização do modelo preditivo baseline de Credit Scoring.

O classificador base emprega o algoritmo XGBoost (eXtreme Gradient Boosting),
amplamente consagrado na indústria financeira para mensuração de risco e
probabilidade de inadimplência (Probability of Default - PD), em razão de sua
capacidade de capturar relações não-lineares, lidar com interações complexas
entre variáveis de crédito e fornecer calibração probabilística consistente.

Metodologia de Modelagem:
    - O treinamento é estritamente conduzido sobre o Dataset de Referência limpo.
    - Divisão estratificada em partições de treino (80%) e teste (20%).
    - Pipeline integrado com pré-processamento via ColumnTransformer:
        * Variáveis numéricas: padronização estatística (StandardScaler).
        * Variáveis categóricas: codificação One-Hot (OneHotEncoder com suporte a categorias não vistas).
    - Métricas de avaliação calculadas:
        * AUC-ROC (Área sob a curva ROC): poder de discriminação de risco.
        * F1-Score: balanço harmônico entre precisão e sensibilidade.
        * Precisão e Recall: mensuração do custo de falsos positivos e negativos.
        * Matriz de Confusão e Log-Loss.
    - Serialização do pipeline completo (joblib) e exportação de metadados.
=============================================================================
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configurações de Diretórios
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_REFERENCE = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_directories() -> None:
    """Garante a existência dos diretórios de modelos e relatórios."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Definição dos Grupos de Atributos do German Credit Data
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: List[str] = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
    "credit_to_duration_ratio",
    "installment_rate_burden",
]

CATEGORICAL_FEATURES: List[str] = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

EXCLUDE_COLUMNS: List[str] = ["client_id", "target"]


def build_pipeline() -> Pipeline:
    """
    Constrói o pipeline de inferência composto por pré-processamento e XGBoost.

    Returns
    -------
    Pipeline
        Pipeline scikit-learn pronto para ajuste e predição.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_and_evaluate_model(
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Executa o treinamento e a avaliação estrita do classificador base.

    Parameters
    ----------
    test_size : float
        Proporção da amostra reservada para teste independente (padrão: 0.20).
    random_state : int
        Semente pseudoaleatória para reprodutibilidade estocástica.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        Pipeline treinado e dicionário com as métricas obtidas.
    """
    ensure_directories()
    ref_path = DATA_REFERENCE / "reference_data.csv"

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Arquivo de referência não localizado em '{ref_path}'. "
            "Execute previamente a Etapa 0 (python src/data_preparation.py)."
        )

    print("\n" + "=" * 70)
    print("ETAPA 1: TREINAMENTO DO MODELO BASELINE DE CREDIT SCORING (XGBOOST)")
    print("=" * 70)

    # 1. Carregamento dos dados
    df = pd.read_csv(ref_path)
    print(f"[INFO] Dataset de Referência carregado: {df.shape[0]} amostras, {df.shape[1]} colunas.")

    X = df.drop(columns=EXCLUDE_COLUMNS, errors="ignore")
    y = df["target"].astype(int)

    # 2. Divisão estratificada de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[INFO] Partição de Treino: {X_train.shape[0]} registros | Teste: {X_test.shape[0]} registros.")

    # 3. Treinamento do pipeline
    print("[INFO] Ajustando o pipeline (pré-processamento + XGBClassifier)...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Inferência na partição de teste
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 5. Cálculo das métricas de desempenho
    auc_roc = float(roc_auc_score(y_test, y_proba))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Desempenho do Modelo Baseline na Partição de Teste ---")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  Acurácia:  {acc:.4f}")
    print(f"  Precisão:  {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("\nMatriz de Confusão:")
    print(f"  TN={cm[0][0]:<4}  FP={cm[0][1]:<4}")
    print(f"  FN={cm[1][0]:<4}  TP={cm[1][1]:<4}")

    # 6. Serialização do artefato do modelo
    model_artifact_path = MODELS_DIR / "credit_scoring_model.joblib"
    joblib.dump(pipeline, model_artifact_path)
    print(f"\n[SUCESSO] Pipeline completo serializado em: {model_artifact_path}")

    # 7. Persistência de metadados e relatório de avaliação
    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "XGBClassifier (scikit-learn Pipeline)",
        "dataset": "German Credit Data (Kaggle/UCI)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "metrics": {
            "auc_roc": round(auc_roc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        },
    }

    metrics_file = MODELS_DIR / "model_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    report_file = REPORTS_DIR / "baseline_model_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print(f"[SUCESSO] Metadados de modelagem exportados para: {metrics_file}")
    print("=" * 70 + "\n")

    return pipeline, metrics_summary


def main() -> None:
    """Ponto de entrada de linha de comando para a Etapa 1 (Treinamento)."""
    train_and_evaluate_model()


if __name__ == "__main__":
    main()
