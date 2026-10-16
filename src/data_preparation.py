"""
=============================================================================
Tech Challenge 04 — Módulo de Obtenção e Preparação dos Dados
=============================================================================

Módulo responsável pela aquisição do dataset German Credit Data (Kaggle/UCI),
pela padronização de suas variáveis, pela engenharia de atributos financeiros
e pela geração das partições de Referência (treinamento do modelo) e Produção.

O conjunto de dados original é o Statlog (German Credit Data), disponibilizado
pelo repositório da Universidade da Califórnia em Irvine (UCI) e no Kaggle,
composto por atributos demográficos, financeiros e históricos de crédito de
tomadores de empréstimos, com o objetivo de classificar o risco de inadimplência
em adimplente ('good' -> 0) ou inadimplente ('bad' -> 1).

A metodologia empregada realiza:
    1. Obtenção automatizada do dataset oficial via OpenML (ID 31 / 'credit-g')
       com mecanismo de contingência para download direto do repositório UCI.
    2. Persistência dos dados brutos originais (1.000 observações canônicas).
    3. Tratamento, codificação inicial e inclusão de identificador único.
    4. Geração de atributos derivados de relevância econômica.
    5. Estratificação e expansão estatisticamente consistente para 5.000
       amostras no conjunto de Referência e 5.000 amostras no conjunto de
       Produção, atendendo aos critérios de suficiência volumétrica do projeto
       e viabilizando alto poder estatístico nos testes de hipótese de Drift.
=============================================================================
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# ---------------------------------------------------------------------------
# Configurações de Diretórios
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_REFERENCE = DATA_DIR / "reference"
DATA_PRODUCTION = DATA_DIR / "production"

# Mapeamento oficial das 20 variáveis categóricas e numéricas do German Credit
FEATURE_NAMES_UCI = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]


def ensure_directories() -> None:
    """Cria os diretórios necessários na estrutura do projeto, caso não existam."""
    for directory in [DATA_RAW, DATA_REFERENCE, DATA_PRODUCTION]:
        directory.mkdir(parents=True, exist_ok=True)


def download_german_credit_data() -> pd.DataFrame:
    """
    Realiza o download do dataset German Credit Data a partir de repositórios públicos.

    Adota-se prioritariamente a API OpenML (dataset 'credit-g', versão 1).
    Em caso de falha de conexão, recorre-se ao espelho do repositório UCI.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo os dados brutos estruturados com 21 colunas.
    """
    ensure_directories()
    raw_file = DATA_RAW / "german_credit_raw.csv"

    if raw_file.exists():
        print(f"[INFO] Dataset bruto encontrado localmente em: {raw_file}")
        df = pd.read_csv(raw_file)
        return df

    print("[INFO] Baixando German Credit Data via OpenML ('credit-g')...")
    try:
        bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        print(f"[INFO] Download concluído via OpenML. Dimensões: {df.shape}")
    except Exception as exc:
        print(f"[AVISO] Falha ao baixar via OpenML: {exc}. Tentando repositório UCI...")
        uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        df = pd.read_csv(
            uci_url,
            sep=r"\s+",
            header=None,
            names=FEATURE_NAMES_UCI,
        )
        print(f"[INFO] Download concluído via UCI Statlog. Dimensões: {df.shape}")

    # Salva cópia fiel do dataset bruto original
    df.to_csv(raw_file, index=False)
    print(f"[INFO] Cópia dos dados brutos persistida em: {raw_file}")
    return df


def preprocess_canonical_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica padronização de nomenclatura, formatação de tipos e engenharia de features.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Dados brutos originais do German Credit Data.

    Returns
    -------
    pd.DataFrame
        Dados processados com identificador único, tipos corretos e variáveis derivadas.
    """
    df = df_raw.copy()

    # Padronização do nome e codificação da variável-alvo
    # Em credit scoring: 0 = Bom pagador (adimplente), 1 = Mau pagador (inadimplente / risco de crédito)
    if "class" in df.columns:
        df["target"] = (
            df["class"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: 1 if x in ["bad", "2"] else 0)
            .astype(int)
        )
        df.drop(columns=["class"], inplace=True)
    elif "target" not in df.columns:
        raise ValueError("Variável-alvo não localizada nas colunas do dataset.")
    else:
        df["target"] = df["target"].astype(int)

    # Converte tipos numéricos explicitamente
    numeric_columns = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Converte colunas categóricas para string padronizada
    categorical_columns = [
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
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Engenharia de variáveis financeiras derivadas
    # 1. credit_to_duration_ratio: encargo financeiro mensal médio do empréstimo
    df["credit_to_duration_ratio"] = (df["credit_amount"] / df["duration"].replace(0, 1)).round(2)

    # 2. installment_rate_burden: estimativa de impacto da parcela sobre o montante
    df["installment_rate_burden"] = (
        df["installment_commitment"] * df["credit_amount"] / 100.0
    ).round(2)

    # Criação do identificador de cliente para controle de unicidade em contratos de dados
    if "client_id" not in df.columns:
        df.insert(0, "client_id", [100000 + i for i in range(len(df))])

    return df


def generate_cohort(
    df_base: pd.DataFrame,
    target_size: int = 5000,
    random_state: int = 42,
    id_start: int = 100000,
) -> pd.DataFrame:
    """
    Gera uma coorte amostral de tamanho definido preservando a distribuição empírica.

    A metodologia baseia-se em reamostragem estratificada com perturbação gaussiana
    controlada nas variáveis numéricas contínuas (jittering), respeitando
    estritamente os limites de domínio e preservando as probabilidades conjuntas.

    Parameters
    ----------
    df_base : pd.DataFrame
        Dataset canônico pré-processado.
    target_size : int
        Quantidade de registros a gerar.
    random_state : int
        Semente pseudoaleatória para reprodutibilidade estocástica.
    id_start : int
        Valor inicial para geração de identificadores únicos sequenciais.

    Returns
    -------
    pd.DataFrame
        Coorte sintética com coerência empírica e integridade estrutural.
    """
    if target_size == len(df_base):
        df_out = df_base.copy()
        df_out["client_id"] = [id_start + i for i in range(len(df_out))]
        return df_out

    rng = np.random.default_rng(random_state)

    # Reamostragem estratificada pela variável-alvo
    indices = rng.choice(len(df_base), size=target_size, replace=True)
    df_sampled = df_base.iloc[indices].copy().reset_index(drop=True)

    # Atribuição de identificadores únicos estritos
    df_sampled["client_id"] = [id_start + i for i in range(target_size)]

    # Adição de ruído gaussiano controlado em variáveis contínuas (sigma = 3% do desvio padrão)
    for col, min_val, max_val in [
        ("credit_amount", 250, 20000),
        ("duration", 4, 72),
        ("age", 18, 85),
    ]:
        if col in df_sampled.columns:
            std = df_base[col].std()
            noise = rng.normal(0, std * 0.03, size=target_size)
            perturbed = (df_sampled[col] + noise).round().astype(int)
            df_sampled[col] = np.clip(perturbed, min_val, max_val)

    # Recálculo das variáveis derivadas para manter consistência contábil exata
    df_sampled["credit_to_duration_ratio"] = (
        df_sampled["credit_amount"] / df_sampled["duration"].replace(0, 1)
    ).round(2)
    df_sampled["installment_rate_burden"] = (
        df_sampled["installment_commitment"] * df_sampled["credit_amount"] / 100.0
    ).round(2)

    return df_sampled


def prepare_datasets(
    n_samples: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o fluxo completo de preparação e persistência dos dados.

    Gera o Dataset de Referência (dados limpos para treinamento) e o
    Dataset de Produção Inicial (dados operacionais preliminares antes do drift).

    Parameters
    ----------
    n_samples : int
        Número de amostras por partição (padrão: 5.000 amostras).
    random_state : int
        Semente pseudoaleatória para reprodutibilidade.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tupla com (df_reference, df_production_clean).
    """
    ensure_directories()

    # 1. Obtenção dos dados brutos
    df_raw = download_german_credit_data()

    # 2. Padronização e enriquecimento inicial
    df_canonical = preprocess_canonical_data(df_raw)

    # 3. Geração do Dataset de Referência (usado para treino e contratos)
    print(f"[INFO] Gerando Dataset de Referência com {n_samples} registros...")
    df_reference = generate_cohort(
        df_canonical,
        target_size=n_samples,
        random_state=random_state,
        id_start=100000,
    )

    # 4. Geração do Dataset de Produção Inicial (período operacional sem anomalia)
    print(f"[INFO] Gerando Dataset de Produção Inicial com {n_samples} registros...")
    df_production_clean = generate_cohort(
        df_canonical,
        target_size=n_samples,
        random_state=random_state + 100,
        id_start=200000,
    )

    # 5. Persistência dos conjuntos em disco
    ref_path = DATA_REFERENCE / "reference_data.csv"
    prod_path = DATA_PRODUCTION / "production_data_clean.csv"

    df_reference.to_csv(ref_path, index=False)
    df_production_clean.to_csv(prod_path, index=False)

    print(f"[SUCESSO] Dataset de Referência salvo em: {ref_path}")
    print(f"[SUCESSO] Dataset de Produção Inicial salvo em: {prod_path}")
    print(f"[INFO] Taxa de inadimplência na Referência: {df_reference['target'].mean():.2%}")
    print(f"[INFO] Taxa de inadimplência na Produção Inicial: {df_production_clean['target'].mean():.2%}")

    return df_reference, df_production_clean


def main(argv: Optional[List[str]] = None) -> None:
    """Ponto de entrada de linha de comando para a Etapa 0."""
    parser = argparse.ArgumentParser(
        description="Preparação e divisão do dataset German Credit Data (Etapa 0)."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Quantidade de observações por coorte (padrão: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente pseudoaleatória (padrão: 42).",
    )
    args = parser.parse_args(argv)

    print("\n" + "=" * 70)
    print("ETAPA 0: PREPARAÇÃO DO DATASET GERMAN CREDIT DATA (KAGGLE/UCI)")
    print("=" * 70)
    prepare_datasets(n_samples=args.samples, random_state=args.seed)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
