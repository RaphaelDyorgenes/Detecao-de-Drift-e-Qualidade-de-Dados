"""
=============================================================================
Tech Challenge 04 — Etapa 2: Simulação e Detecção Estatística de Drift
=============================================================================

Módulo responsável pela simulação controlada de degradação temporal (Data Drift
e Concept Drift), geração de inferências em ambiente operacional e detecção
estatística formal das mudanças distribucionais utilizando o Evidently AI e
testes de hipótese paramétricos e não-paramétricos (Kolmogorov-Smirnov,
Population Stability Index - PSI e Qui-Quadrado).

Fundamentação do Cenário de Risco (Contexto Econômico):
    Em operações de crédito consignado e financiamento bancário, choques
    macroeconômicos (como surtos inflacionários e elevação das taxas de juros)
    degradam silenciosamente os escores preditivos. Os tomadores passam a
    demandar empréstimos mais elevados (para compensar a perda do poder de compra)
    e com prazos mais longos (para diluir o impacto das parcelas no orçamento
    mensal), ao mesmo tempo em que os saldos de conta corrente sofrem deterioração.

Variáveis Objeto de Alteração Controlada (Mínimo de duas exigidas):
    1. credit_amount (Data Drift Numérico): Deslocamento da distribuição para a
       direita com ampliação da cauda de valores elevados (+45% a +60%).
    2. duration (Data Drift Numérico): Aumento significativo no prazo médio de
       amortização dos contratos (+35% a +50%).
    3. checking_status (Data Drift Categórico): Elevação acentuada na frequência
       de clientes com saldo devedor (< 0 DM) e redução de saldos superiores.
    4. Concept Drift: Aumento na probabilidade condicional de inadimplência (P(Y|X))
       entre proponentes com alto endividamento e ausência de reserva de liquidez.

Entregáveis Deste Módulo:
    - Dataset de Produção persistido com simulação de degradação temporal.
    - Colunas de inferência anexadas (predição de classe e escore probabilístico).
    - Relatório analítico interativo do Evidently AI (data_drift_report.html).
    - Matriz de testes estatísticos comprovando formalmente as variáveis degradadas.
=============================================================================
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats

# Evidently AI (versão 0.7+)
from evidently import Report
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------------------------
# Configurações de Diretórios
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_REFERENCE = PROJECT_ROOT / "data" / "reference"
DATA_PRODUCTION = PROJECT_ROOT / "data" / "production"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_directories() -> None:
    """Garante a existência das pastas necessárias para artefatos e relatórios."""
    DATA_PRODUCTION.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Funções de Simulação de Data Drift e Concept Drift
# =============================================================================

def simulate_economic_drift(
    df_clean: pd.DataFrame,
    intensity: str = "moderate",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simula Data Drift e Concept Drift representando um choque macroeconômico.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Dataset original de produção sem anomalias distribucionais.
    intensity : str
        Intensidade do choque: 'mild', 'moderate' ou 'severe'.
    random_state : int
        Semente pseudoaleatória para reprodutibilidade.

    Returns
    -------
    pd.DataFrame
        Dataset de Produção com distribuição alterada e variáveis derivadas recalculadas.
    """
    df = df_clean.copy()
    rng = np.random.default_rng(random_state)
    n = len(df)

    multipliers = {
        "mild": {"credit_factor": 1.25, "duration_add": 6, "checking_bias": 0.20},
        "moderate": {"credit_factor": 1.45, "duration_add": 12, "checking_bias": 0.40},
        "severe": {"credit_factor": 1.70, "duration_add": 18, "checking_bias": 0.60},
    }
    cfg = multipliers.get(intensity, multipliers["moderate"])

    print(f"\n[SIMULAÇÃO] Aplicando choque de drift econômico (intensidade: '{intensity}')...")

    # -------------------------------------------------------------------------
    # 1. DRIFT EM 'credit_amount': Elevação do montante solicitado (efeito inflacionário)
    # -------------------------------------------------------------------------
    credit_noise = rng.normal(0, df["credit_amount"].std() * 0.10, size=n)
    drifted_credit = (df["credit_amount"] * cfg["credit_factor"] + credit_noise).round().astype(int)
    # Limitação dentro dos limites plausíveis de mercado
    df["credit_amount"] = np.clip(drifted_credit, 300, 25000)
    print(
        f"  - credit_amount alterado: Média passou de {df_clean['credit_amount'].mean():.1f} "
        f"para {df['credit_amount'].mean():.1f} (acréscimo de +{(df['credit_amount'].mean()/df_clean['credit_amount'].mean()-1)*100:.1f}%)."
    )

    # -------------------------------------------------------------------------
    # 2. DRIFT EM 'duration': Alongamento dos prazos de amortização
    # -------------------------------------------------------------------------
    duration_noise = rng.integers(-2, 5, size=n)
    drifted_duration = (df["duration"] + cfg["duration_add"] + duration_noise).astype(int)
    df["duration"] = np.clip(drifted_duration, 6, 72)
    print(
        f"  - duration alterada: Média passou de {df_clean['duration'].mean():.1f} "
        f"para {df['duration'].mean():.1f} meses (acréscimo de +{(df['duration'].mean()/df_clean['duration'].mean()-1)*100:.1f}%)."
    )

    # -------------------------------------------------------------------------
    # 3. DRIFT CATEGÓRICO EM 'checking_status': Deterioração de saldos em conta
    # -------------------------------------------------------------------------
    # Proporção de clientes com saldo devedor (< 0) é inflacionada artificialmente
    mask_to_deteriorate = rng.random(size=n) < cfg["checking_bias"]
    df.loc[mask_to_deteriorate, "checking_status"] = "<0"
    print(
        f"  - checking_status alterado: Frequência de '<0' (saldo devedor) passou de "
        f"{(df_clean['checking_status'] == '<0').mean():.1%} para {(df['checking_status'] == '<0').mean():.1%}."
    )

    # -------------------------------------------------------------------------
    # 4. Recálculo das variáveis financeiras derivadas
    # -------------------------------------------------------------------------
    df["credit_to_duration_ratio"] = (df["credit_amount"] / df["duration"].replace(0, 1)).round(2)
    df["installment_rate_burden"] = (
        df["installment_commitment"] * df["credit_amount"] / 100.0
    ).round(2)

    # -------------------------------------------------------------------------
    # 5. CONCEPT DRIFT: Aumento da inadimplência em tomadores vulneráveis
    # -------------------------------------------------------------------------
    # Em cenários de crise, clientes com dívida elevada e saldo negativo têm risco amplificado
    high_risk_condition = (df["credit_amount"] > df["credit_amount"].median()) & (
        df["checking_status"] == "<0"
    )
    # Aplica default induzido em 60% dessa parcela
    elevated_defaults = rng.random(size=n) < 0.60
    df.loc[high_risk_condition & elevated_defaults, "target"] = 1
    print(
        f"  - Concept Drift simulado: Taxa real de inadimplência passou de "
        f"{df_clean['target'].mean():.1%} para {df['target'].mean():.1%}."
    )

    return df


# =============================================================================
# Métodos Estatísticos de Detecção de Drift
# =============================================================================

def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calcula o Population Stability Index (PSI) entre duas distribuições contínuas.

    O PSI avalia a distância entre distribuições empíricas particionadas em decis:
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))

    Critérios de Interpretação:
        - PSI < 0.10: Estabilidade (sem alteração significativa).
        - 0.10 <= PSI < 0.25: Mudança moderada (requer monitoramento).
        - PSI >= 0.25: Mudança substancial (Drift severo / retreinamento recomendado).

    Parameters
    ----------
    expected : pd.Series
        Série da partição de Referência (esperada).
    actual : pd.Series
        Série da partição de Produção (observada).
    num_bins : int
        Número de decis/quantis para discretização.
    epsilon : float
        Fator de suavização para evitar divisão por zero ou logaritmo nulo.

    Returns
    -------
    float
        Valor escalar do PSI.
    """
    # Determinação dos quantis a partir da distribuição de referência
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(expected.dropna(), quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Contagem de frequências
    expected_counts = np.histogram(expected.dropna(), bins=bin_edges)[0]
    actual_counts = np.histogram(actual.dropna(), bins=bin_edges)[0]

    # Proporções relativas
    expected_pct = (expected_counts / len(expected.dropna())) + epsilon
    actual_pct = (actual_counts / len(actual.dropna())) + epsilon

    # Normalização
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    # Cálculo da soma ponderada
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


def calculate_ks_test(
    reference: pd.Series,
    current: pd.Series,
) -> Tuple[float, float, bool]:
    """
    Aplica o teste bicaudal de duas amostras de Kolmogorov-Smirnov (KS-test).

    Testa a hipótese nula H0: Ambas as amostras provêm da mesma distribuição contínua.
    Se p-valor < 0.05, rejeita-se H0, confirmando a ocorrência de Data Drift estatístico.

    Parameters
    ----------
    reference : pd.Series
        Amostra da base de referência.
    current : pd.Series
        Amostra da base de produção.

    Returns
    -------
    Tuple[float, float, bool]
        Estatística KS (distância D máxima), p-valor e booleano de confirmação de drift.
    """
    res = stats.ks_2samp(reference.dropna(), current.dropna())
    ks_stat = float(res.statistic)
    p_value = float(res.pvalue)
    drift_detected = p_value < 0.05
    return ks_stat, p_value, drift_detected


def calculate_categorical_chi2(
    reference: pd.Series,
    current: pd.Series,
) -> Tuple[float, float, bool]:
    """
    Aplica o teste Qui-Quadrado de homogeneidade para variáveis categóricas.

    Parameters
    ----------
    reference : pd.Series
        Frequências da base de referência.
    current : pd.Series
        Frequências da base de produção.

    Returns
    -------
    Tuple[float, float, bool]
        Estatística Chi-2, p-valor e indicador booleano de drift.
    """
    # Alinhamento das categorias
    all_categories = sorted(list(set(reference.dropna().unique()).union(set(current.dropna().unique()))))
    ref_counts = reference.value_counts().reindex(all_categories, fill_value=0)
    curr_counts = current.value_counts().reindex(all_categories, fill_value=0)

    contingency_table = np.array([ref_counts.values, curr_counts.values])
    # Adiciona suavização caso haja zero
    contingency_table = contingency_table + 1e-3

    res = stats.chi2_contingency(contingency_table)
    chi2_stat = float(res.statistic)
    p_value = float(res.pvalue)
    drift_detected = p_value < 0.05
    return chi2_stat, p_value, drift_detected


# =============================================================================
# Execução dos Testes e Geração do Relatório Evidently AI
# =============================================================================

def execute_statistical_audit(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Executa bateria exaustiva de testes estatísticos para todas as variáveis do modelo.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Dataset de Referência.
    df_prod : pd.DataFrame
        Dataset de Produção com drift.

    Returns
    -------
    Dict[str, Any]
        Dicionário estruturado com os resultados de cada feature e sumário executivo.
    """
    numeric_cols = [
        "credit_amount",
        "duration",
        "age",
        "credit_to_duration_ratio",
        "installment_commitment",
        "residence_since",
        "existing_credits",
        "num_dependents",
        "installment_rate_burden",
    ]

    categorical_cols = [
        "checking_status",
        "savings_status",
        "housing",
        "foreign_worker",
        "credit_history",
        "purpose",
        "employment",
        "personal_status",
        "property_magnitude",
        "job",
    ]

    results = {}
    drift_count = 0

    print("\n" + "=" * 75)
    print("PAINEL DE AUDITORIA ESTATÍSTICA DE DATA DRIFT (KS, PSI E QUI-QUADRADO)")
    print("=" * 75)
    print(f"{'Variável':<26} {'Tipo':<12} {'Teste':<8} {'Estatística':<12} {'p-valor':<10} {'Status':<16}")
    print("-" * 75)

    # 1. Avaliação de Variáveis Numéricas
    for col in numeric_cols:
        if col in df_ref.columns and col in df_prod.columns:
            ks_stat, p_val, ks_drift = calculate_ks_test(df_ref[col], df_prod[col])
            psi_val = calculate_psi(df_ref[col], df_prod[col])

            status_str = "DRIFT DETECTADO" if (ks_drift and psi_val >= 0.10) else "ESTÁVEL"
            if status_str == "DRIFT DETECTADO":
                drift_count += 1

            results[col] = {
                "type": "numeric",
                "test": "KS + PSI",
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": p_val,
                "psi_value": round(psi_val, 4),
                "psi_interpretation": (
                    "Substancial (PSI >= 0.25)"
                    if psi_val >= 0.25
                    else ("Moderado (0.10 <= PSI < 0.25)" if psi_val >= 0.10 else "Estável (PSI < 0.10)")
                ),
                "drift_detected": bool(ks_drift and psi_val >= 0.10),
            }

            p_val_fmt = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            print(f"{col:<26} {'Numérico':<12} {'KS':<8} {ks_stat:<12.4f} {p_val_fmt:<10} {status_str:<16}")

    # 2. Avaliação de Variáveis Categóricas
    for col in categorical_cols:
        if col in df_ref.columns and col in df_prod.columns:
            chi2_stat, p_val, chi2_drift = calculate_categorical_chi2(df_ref[col], df_prod[col])
            status_str = "DRIFT DETECTADO" if chi2_drift else "ESTÁVEL"
            if chi2_drift:
                drift_count += 1

            results[col] = {
                "type": "categorical",
                "test": "Chi-Square",
                "chi2_statistic": round(chi2_stat, 4),
                "p_value": p_val,
                "drift_detected": bool(chi2_drift),
            }

            p_val_fmt = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            print(f"{col:<26} {'Categórico':<12} {'Chi2':<8} {chi2_stat:<12.4f} {p_val_fmt:<10} {status_str:<16}")

    print("=" * 75)
    print(f"Total de variáveis monitoradas: {len(results)} | Com Drift Confirmado: {drift_count}")
    print("=" * 75 + "\n")

    return {
        "timestamp": datetime.now().isoformat(),
        "total_features_evaluated": len(results),
        "total_drifted_features": drift_count,
        "features": results,
    }


def generate_evidently_report(
    df_ref: pd.DataFrame,
    df_prod: pd.DataFrame,
    output_html_path: Path,
    output_json_path: Path,
) -> None:
    """
    Gera e persiste os relatórios visual (HTML) e analítico (JSON) via Evidently AI.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Dataset de Referência.
    df_prod : pd.DataFrame
        Dataset de Produção com drift.
    output_html_path : Path
        Caminho de destino do relatório HTML.
    output_json_path : Path
        Caminho de destino do relatório JSON.
    """
    print("[INFO] Gerando Relatório de Observabilidade com Evidently AI (DataDriftPreset)...")

    # Colunas excluídas da comparação de drift (identificador transacional)
    cols_to_compare = [c for c in df_ref.columns if c not in ["client_id"]]
    ref_subset = df_ref[cols_to_compare].copy()
    prod_subset = df_prod[cols_to_compare].copy()

    # Criação do relatório Evidently
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_subset, current_data=prod_subset)

    # Persistência do relatório interativo HTML
    snapshot.save_html(str(output_html_path))
    print(f"[SUCESSO] Relatório HTML do Evidently gerado com sucesso em: {output_html_path}")

    # Persistência dos resultados brutos em JSON
    snapshot.save_json(str(output_json_path))
    print(f"[SUCESSO] Sumário JSON do Evidently persistido em: {output_json_path}")


# =============================================================================
# Pipeline Principal da Etapa 2
# =============================================================================

def run_drift_pipeline(
    drift_intensity: str = "moderate",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executa o fluxo completo da Etapa 2:
        1. Carrega dados de Referência e Produção inicial.
        2. Carrega modelo preditivo treinado na Etapa 1.
        3. Aplica o choque econômico (Data Drift e Concept Drift).
        4. Gera inferências probabilísticas e rótulos preditos para o novo dataset.
        5. Conduz bateria de testes estatísticos (KS, PSI, Qui-Quadrado).
        6. Gera e salva os relatórios HTML e JSON do Evidently AI.

    Parameters
    ----------
    drift_intensity : str
        Grau de alteração nos dados: 'mild', 'moderate' ou 'severe'.
    random_state : int
        Semente pseudoaleatória.

    Returns
    -------
    Dict[str, Any]
        Sumário executivo do pipeline de detecção de drift.
    """
    ensure_directories()

    ref_path = DATA_REFERENCE / "reference_data.csv"
    prod_clean_path = DATA_PRODUCTION / "production_data_clean.csv"
    model_path = MODELS_DIR / "credit_scoring_model.joblib"

    # Verificação de pré-requisitos
    for path, step in [
        (ref_path, "Etapa 0 (data_preparation.py)"),
        (prod_clean_path, "Etapa 0 (data_preparation.py)"),
        (model_path, "Etapa 1 (model_training.py)"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Artefato obrigatório '{path}' não encontrado. Execute previamente a {step}."
            )

    print("\n" + "=" * 70)
    print("ETAPA 2: SIMULAÇÃO E DETECÇÃO DE DRIFT (EVIDENTLY AI & ESTATÍSTICA)")
    print("=" * 70)

    # 1. Carregamento dos dados
    df_reference = pd.read_csv(ref_path)
    df_prod_clean = pd.read_csv(prod_clean_path)
    print(f"[INFO] Dataset de Referência: {len(df_reference)} observações.")
    print(f"[INFO] Dataset de Produção Inicial: {len(df_prod_clean)} observações.")

    # 2. Simulação do Data Drift e Concept Drift
    df_production_drifted = simulate_economic_drift(
        df_prod_clean,
        intensity=drift_intensity,
        random_state=random_state,
    )

    # 3. Geração de Predições em Produção utilizando o modelo baseline
    print("\n[INFERÊNCIA] Aplicando modelo baseline treinado sobre o Dataset de Produção...")
    pipeline = joblib.load(model_path)

    X_prod = df_production_drifted.drop(columns=["client_id", "target"], errors="ignore")
    df_production_drifted["prediction"] = pipeline.predict(X_prod)
    df_production_drifted["prediction_score"] = pipeline.predict_proba(X_prod)[:, 1].round(4)

    # Predições sobre a base de referência (para comparação completa de Target/Prediction Drift)
    X_ref = df_reference.drop(columns=["client_id", "target"], errors="ignore")
    df_reference["prediction"] = pipeline.predict(X_ref)
    df_reference["prediction_score"] = pipeline.predict_proba(X_ref)[:, 1].round(4)

    # Persistência do dataset de produção com drift e predições
    prod_drifted_path = DATA_PRODUCTION / "production_drifted.csv"
    df_production_drifted.to_csv(prod_drifted_path, index=False)
    print(f"[SUCESSO] Dataset de Produção com Drift e Predições salvo em: {prod_drifted_path}")

    # 4. Auditoria Estatística Formal (KS, PSI, Qui-Quadrado)
    statistical_summary = execute_statistical_audit(df_reference, df_production_drifted)
    stat_report_file = REPORTS_DIR / "statistical_drift_tests.json"
    with open(stat_report_file, "w", encoding="utf-8") as f:
        json.dump(statistical_summary, f, indent=2, ensure_ascii=False)
    print(f"[SUCESSO] Auditoria estatística formal persistida em: {stat_report_file}")

    # 5. Emissão dos Relatórios Oficiais com Evidently AI
    evidently_html = REPORTS_DIR / "data_drift_report.html"
    evidently_json = REPORTS_DIR / "data_drift_summary.json"
    generate_evidently_report(
        df_reference,
        df_production_drifted,
        evidently_html,
        evidently_json,
    )

    print("\n" + "=" * 70)
    print("CONCLUSAO DA ETAPA 2")
    print("=" * 70)
    print("  1. Simulação de choque macroeconômico executada com sucesso.")
    print("  2. Variáveis com Drift confirmado:")
    for feature, details in statistical_summary["features"].items():
        if details.get("drift_detected"):
            print(f"     * {feature} ({details.get('test')})")
    print(f"  3. Relatório HTML do Evidently disponível em: {evidently_html}")
    print("=" * 70 + "\n")

    return statistical_summary


def main() -> None:
    """Ponto de entrada de linha de comando para a Etapa 2."""
    run_drift_pipeline()


if __name__ == "__main__":
    main()
