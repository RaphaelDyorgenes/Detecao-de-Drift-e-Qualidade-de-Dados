"""
=============================================================================
Tech Challenge 04 — Bateria de Testes Automatizados (Pytest)
=============================================================================

Suite de testes automatizados para verificação da integridade das 4 etapas
do pipeline de Observabilidade, Detecção de Drift e Governança de Modelos:

    - Testes de Contratos de Dados (Pandera): Aprovação de lote limpo e
      rejeição defensiva obrigatória de lotes corrompidos.
    - Testes do Modelo Baseline (XGBoost): Carregamento, inferência de probabilidades
      e limites mínimos de acurácia/AUC-ROC.
    - Testes de Detecção de Drift (Estatística e Evidently AI): Comprovação
      do diagnóstico de drift nas variáveis alteradas propositalmente.
    - Testes de Observabilidade (MLflow, Prometheus e Dashboard): Verificação
      dos artefatos de telemetria e integridade dos relatórios HTML.
    - Testes de Governança e LGPD: Validação de ausência de vazamento de PII
      e uso estrito de chave sintética pseudonimizada.
=============================================================================
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data_validation import (
    build_credit_contract_schema,
    create_corrupted_dataset_for_testing,
)
from src.governance import audit_pii_and_anonymization, evaluate_fairness_and_bias
from src.observability import collect_pipeline_telemetry
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


@pytest.fixture(scope="session")
def reference_data() -> pd.DataFrame:
    """Fixture que fornece o Dataset de Referência."""
    ref_path = DATA_DIR / "reference" / "reference_data.csv"
    assert ref_path.exists(), f"Dataset de referência não encontrado em: {ref_path}"
    df = pd.read_csv(ref_path)
    return df


@pytest.fixture(scope="session")
def production_data() -> pd.DataFrame:
    """Fixture que fornece o Dataset de Produção com Drift."""
    prod_path = DATA_DIR / "production" / "production_drifted.csv"
    assert prod_path.exists(), f"Dataset de produção não encontrado em: {prod_path}"
    df = pd.read_csv(prod_path)
    return df


# =============================================================================
# 1. TESTES DA ETAPA 1: DATA CONTRACTS (PANDERA)
# =============================================================================


def test_pandera_schema_accepts_clean_reference_data(reference_data: pd.DataFrame):
    """
    Garante que o Dataset de Referência cumpre 100% das regras estritas do contrato
    sem lançar nenhuma SchemaErrors.
    """
    schema = build_credit_contract_schema()
    # Não deve lançar exceção
    validated_df = schema.validate(reference_data)
    assert validated_df is not None
    assert len(validated_df) == len(reference_data)


def test_pandera_schema_rejects_corrupted_batch_strictly(reference_data: pd.DataFrame):
    """
    Garante que o Quality Gate intercepta formalmente dados anômalos
    (menores de idade, montantes negativos, nulos e categorias inválidas).
    """
    import pandera.errors as pa_errors

    schema = build_credit_contract_schema()
    corrupted_df = create_corrupted_dataset_for_testing(reference_data)

    with pytest.raises(pa_errors.SchemaErrors) as exc_info:
        schema.validate(corrupted_df, lazy=True)

    failure_df = exc_info.value.failure_cases
    assert len(failure_df) > 0, "O contrato deveria ter identificado violações"
    failed_columns = failure_df["column"].dropna().unique().tolist()

    # Valida interceptação das colunas corrompidas
    assert any(col in failed_columns for col in ["age", "credit_amount", "checking_status", "housing", "client_id", "target"])


# =============================================================================
# 2. TESTES DA ETAPA 1: MODELO BASELINE (XGBOOST)
# =============================================================================


def test_baseline_model_artifact_exists_and_infers(reference_data: pd.DataFrame):
    """
    Verifica se o artefato serializado existe, é carregável e gera probabilidades
    no intervalo [0, 1].
    """
    model_path = MODELS_DIR / "credit_scoring_model.joblib"
    assert model_path.exists(), "Artefato de modelo não encontrado"

    pipeline = joblib.load(model_path)
    sample = reference_data.drop(columns=["target", "client_id"], errors="ignore").head(50)

    preds = pipeline.predict(sample)
    probs = pipeline.predict_proba(sample)

    assert len(preds) == 50
    assert set(preds).issubset({0, 1})
    assert probs.shape == (50, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_baseline_model_metrics_meet_rubric_standards():
    """
    Garante que as métricas de avaliação registradas para o modelo atendem aos
    requisitos de desempenho para concessão de crédito (AUC-ROC > 0.85).
    """
    metrics_path = REPORTS_DIR / "baseline_model_evaluation.json"
    assert metrics_path.exists(), "Relatório de métricas do modelo não encontrado"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    test_metrics = metrics.get("metrics") or metrics.get("test_metrics", {})
    auc_roc = test_metrics.get("auc_roc", 0.0)
    accuracy = test_metrics.get("accuracy", 0.0)

    assert auc_roc >= 0.85, f"AUC-ROC ({auc_roc}) abaixo do patamar de qualidade mínima (0.85)"
    assert accuracy >= 0.80, f"Acurácia ({accuracy}) abaixo do patamar mínimo (0.80)"


# =============================================================================
# 3. TESTES DA ETAPA 2: DETECÇÃO ESTATÍSTICA DE DRIFT
# =============================================================================


def test_drift_detected_in_credit_amount_and_duration(
    reference_data: pd.DataFrame, production_data: pd.DataFrame
):
    """
    Garante que a simulação de choque macroeconômico provocou drift estatístico
    significativo em 'credit_amount' e 'duration' (KS-test p-valor < 0.01).
    """
    # Teste KS em credit_amount
    ks_amount = stats.ks_2samp(
        reference_data["credit_amount"], production_data["credit_amount"]
    )
    assert ks_amount.pvalue < 0.01, f"credit_amount deveria apresentar drift estatístico (p-valor: {ks_amount.pvalue})"

    # Teste KS em duration
    ks_duration = stats.ks_2samp(
        reference_data["duration"], production_data["duration"]
    )
    assert ks_duration.pvalue < 0.01, f"duration deveria apresentar drift estatístico (p-valor: {ks_duration.pvalue})"


def test_drift_reports_generated():
    """
    Garante que os relatórios exigidos da Etapa 2 foram gerados no diretório reports/.
    """
    html_report = REPORTS_DIR / "data_drift_report.html"
    summary_json = REPORTS_DIR / "data_drift_summary.json"
    stats_json = REPORTS_DIR / "statistical_drift_tests.json"

    assert html_report.exists() and html_report.stat().st_size > 1000, "HTML do Evidently inválido ou vazio"
    assert summary_json.exists(), "Sumário JSON do Evidently não encontrado"
    assert stats_json.exists(), "Auditoria estatística JSON não encontrada"


# =============================================================================
# 4. TESTES DA ETAPA 3: OBSERVABILIDADE E MÉTRICAS
# =============================================================================


def test_observability_telemetry_collection():
    """
    Verifica se a coleta de telemetria consolida métricas de infraestrutura,
    validação e modelos sem erros.
    """
    telemetry = collect_pipeline_telemetry()
    assert "infrastructure" in telemetry
    assert "process_memory_rss_mb" in telemetry["infrastructure"]
    assert "model_performance" in telemetry
    assert "drift_summary" in telemetry


def test_prometheus_metrics_file_validity():
    """
    Garante que o arquivo OpenMetrics/Prometheus foi exportado e contém
    as métricas canônicas de saúde da esteira.
    """
    prom_file = REPORTS_DIR / "prometheus_metrics.prom"
    assert prom_file.exists(), "Arquivo prometheus_metrics.prom não encontrado"

    content = prom_file.read_text(encoding="utf-8")
    assert "pipeline_task_status" in content
    assert "model_auc_roc" in content
    assert "drift_detected_features_total" in content


def test_monitoring_dashboard_html_structure():
    """
    Verifica se o dashboard HTML executivo existe e possui tags essenciais.
    """
    dashboard_file = REPORTS_DIR / "pipeline_monitoring_dashboard.html"
    assert dashboard_file.exists(), "Dashboard executivo HTML não encontrado"

    html = dashboard_file.read_text(encoding="utf-8")
    assert "Observabilidade &amp; Monitoramento Contínuo" in html or "Observabilidade" in html
    assert "Desempenho Modelo (AUC-ROC)" in html
    assert "table" in html


# =============================================================================
# 5. TESTES DA ETAPA 4: GOVERNANÇA, LGPD E FAIRNESS
# =============================================================================


def test_pii_audit_confirms_no_direct_identifiers(reference_data: pd.DataFrame):
    """
    Garante a ausência de dados cadastrais diretos proibidos (CPF, RG, Nome, etc.)
    e validação do uso exclusivo de identificador sintético pseudonimizado.
    """
    audit = audit_pii_and_anonymization(reference_data)
    assert audit["pii_leakage_detected"] is False, "Vazamento de PII detectado no dataset"
    assert audit["surrogate_key_used"] is True, "Dataset não utiliza chave surrogate pseudonimizada"
    assert audit["status"] == "CONFORME"


def test_governance_report_contains_legal_bases():
    """
    Verifica se o relatório formal de governança especifica as bases legais
    da LGPD (Art. 7º, Inciso X) e salvaguardas de revisão automatizada (Art. 20).
    """
    gov_file = REPORTS_DIR / "governance_compliance_report.json"
    assert gov_file.exists(), "Relatório de governança não encontrado"

    with open(gov_file, "r", encoding="utf-8") as f:
        gov = json.load(f)

    assert "legal_bases" in gov
    assert "Art. 7º, Inciso X da LGPD" in gov["legal_bases"]["primary"]["article"]
    assert "automated_decision_safeguards" in gov
