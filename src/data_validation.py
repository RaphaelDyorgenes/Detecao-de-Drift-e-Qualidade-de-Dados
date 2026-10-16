"""
=============================================================================
Tech Challenge 04 — Etapa 1: Validação de Dados e Contratos (Data Contracts)
=============================================================================

Módulo responsável pela especificação, execução e auditoria de contratos
formais de qualidade de dados (Data Contracts) utilizando a biblioteca Pandera.

Na arquitetura de sustentação de Machine Learning em ambientes de produção,
os contratos de dados funcionam como guardiões transacionais (quality gates)
no pipeline de ingestão. Seu propósito consiste em interceptar proativamente
anomalias de esquema, violações de integridade referencial, quebras de tipos
e inconsistências de regras de negócio antes que os registros alcancem as etapas
de inferência ou retreinamento dos modelos estatísticos.

Regras Estritas Implementadas no Contrato:
    1. client_id: Identificador único, não nulo e sem duplicidades.
    2. age: Idade legal obrigatória (>= 18 anos) e limite biológico plausível (<= 100).
    3. credit_amount: Montante financeiro estritamente positivo (> 0) e <= 50.000.
    4. duration: Prazo contratual em meses estritamente positivo (1 a 120 meses).
    5. target: Variável-alvo binária estrita pertencente ao conjunto {0, 1}.
    6. checking_status: Domínio restrito a categorias bancárias homologadas.
    7. savings_status: Domínio restrito a faixas de poupança válidas.
    8. housing: Domínio estrito de situação habitacional ('own', 'rent', 'for free').
    9. foreign_worker: Indicador binário homologado ('yes', 'no').
    10. credit_to_duration_ratio: Consistência matemática contábil (> 0).

Fluxo Operacional:
    - Validação formal do Dataset de Referência (comportamento esperado: APROVADO).
    - Injeção controlada de anomalias sintéticas para teste de estresse defensivo.
    - Validação do lote corrompido (comportamento esperado: REJEITADO com interceptação).
    - Emissão de relatórios estruturados de auditoria em console e arquivo JSON.
=============================================================================
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.pandas import Check, Column, DataFrameSchema

# Desativa alertas redundantes de importação
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

# ---------------------------------------------------------------------------
# Configurações de Diretórios
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_REFERENCE = PROJECT_ROOT / "data" / "reference"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_reports_dir() -> None:
    """Garante a existência do diretório de relatórios do projeto."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Definição Formal do Contrato de Dados (Pandera Schema)
# =============================================================================

def build_credit_contract_schema() -> DataFrameSchema:
    """
    Constrói a especificação formal do contrato de dados para o German Credit Data.

    O esquema estabelece verificações univariadas de tipo, nulidade, unicidade
    e regras determinísticas de negócio aplicáveis a cada coluna do pipeline.

    Returns
    -------
    pa.DataFrameSchema
        Esquema Pandera configurado com validação preguiçosa (lazy evaluation).
    """
    valid_checking = ["<0", "0<=X<200", ">=200", "no checking"]
    valid_savings = ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"]
    valid_housing = ["own", "rent", "for free"]
    valid_foreign = ["yes", "no"]

    schema = DataFrameSchema(
        columns={
            # 1. Identificador único do tomador
            "client_id": Column(
                dtype="int64",
                nullable=False,
                unique=True,
                checks=[
                    Check.greater_than(0, error="O identificador client_id deve ser estritamente positivo."),
                ],
                description="Identificador transacional unívoco do cliente bancário.",
            ),

            # 2. Idade do cliente (requisito de maioridade legal e coerência etária)
            "age": Column(
                dtype="int64",
                nullable=False,
                checks=[
                    Check.greater_than_or_equal_to(18, error="Violação de maioridade legal: age < 18 anos."),
                    Check.less_than_or_equal_to(100, error="Inconsistência cadastral: age > 100 anos."),
                ],
                description="Idade cronológica do solicitante em anos completos.",
            ),

            # 3. Montante financeiro do crédito concedido
            "credit_amount": Column(
                dtype="int64",
                nullable=False,
                checks=[
                    Check.greater_than(0, error="O valor do crédito deve ser estritamente positivo."),
                    Check.less_than_or_equal_to(50000, error="O valor do crédito excede o teto operacional de 50.000."),
                ],
                description="Montante financeiro nominal do empréstimo solicitado.",
            ),

            # 4. Prazo de amortização do empréstimo
            "duration": Column(
                dtype="int64",
                nullable=False,
                checks=[
                    Check.greater_than_or_equal_to(1, error="A duração do contrato deve ser de no mínimo 1 mês."),
                    Check.less_than_or_equal_to(120, error="A duração excede o limite contratual máximo de 120 meses."),
                ],
                description="Duração da operação de crédito expressa em meses.",
            ),

            # 5. Status da conta corrente do cliente
            "checking_status": Column(
                dtype="str",
                nullable=False,
                checks=[
                    Check.isin(valid_checking, error=f"Status de conta inválido. Domínios aceitos: {valid_checking}"),
                ],
                description="Saldo da conta corrente existente do cliente na instituição.",
            ),

            # 6. Status da conta de poupança / investimentos
            "savings_status": Column(
                dtype="str",
                nullable=False,
                checks=[
                    Check.isin(valid_savings, error=f"Status de poupança inválido. Domínios aceitos: {valid_savings}"),
                ],
                description="Faixa de saldo em contas de poupança e aplicações.",
            ),

            # 7. Taxa de comprometimento da parcela sobre a renda disponível
            "installment_commitment": Column(
                dtype="int64",
                nullable=False,
                checks=[
                    Check.in_range(1, 4, error="Taxa de comprometimento da parcela deve situar-se entre 1 e 4."),
                ],
                description="Percentual de comprometimento da renda líquida com a prestação.",
            ),

            # 8. Situação habitacional do proponente
            "housing": Column(
                dtype="str",
                nullable=False,
                checks=[
                    Check.isin(valid_housing, error=f"Situação de moradia inválida. Domínios aceitos: {valid_housing}"),
                ],
                description="Regime habitacional do solicitante ('own', 'rent', 'for free').",
            ),

            # 9. Indicador de trabalhador estrangeiro
            "foreign_worker": Column(
                dtype="str",
                nullable=False,
                checks=[
                    Check.isin(valid_foreign, error=f"Indicador de trabalhador estrangeiro deve ser {valid_foreign}."),
                ],
                description="Classificação sobre condição de trabalhador estrangeiro.",
            ),

            # 10. Variável-alvo binária de solvência/risco
            "target": Column(
                dtype="int64",
                nullable=False,
                checks=[
                    Check.isin([0, 1], error="A variável-alvo deve conter unicamente valores binários 0 ou 1."),
                ],
                description="Indicador de adimplência (0: bom pagador, 1: inadimplente).",
            ),

            # 11. Razão contábil entre crédito e prazo (métrica de parcelamento mensal)
            "credit_to_duration_ratio": Column(
                dtype="float64",
                nullable=False,
                checks=[
                    Check.greater_than(0, error="A razão crédito/duração deve ser estritamente positiva."),
                ],
                description="Razão financeira entre montante concedido e prazo em meses.",
            ),

            # 12. Estimativa do encargo da parcela
            "installment_rate_burden": Column(
                dtype="float64",
                nullable=False,
                checks=[
                    Check.greater_than(0, error="O encargo nominal da parcela deve ser estritamente positivo."),
                ],
                description="Estimativa do valor nominal da parcela amortizada.",
            ),
        },
        coerce=True,
        strict=False,  # Permite presença das demais colunas contextuais (ex.: purpose, job)
    )

    return schema


# =============================================================================
# Execução da Validação e Registro de Auditoria
# =============================================================================

def validate_dataset(
    df: pd.DataFrame,
    schema: DataFrameSchema,
    dataset_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Executa a validação formal de um DataFrame frente ao contrato de dados estabelecido.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset a ser submetido à auditoria de qualidade.
    schema : DataFrameSchema
        Contrato formal com as regras estabelecidas pelo Pandera.
    dataset_name : str
        Rótulo identificador do conjunto de dados para relatório.

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        Tupla com status booleano de aprovação (True se íntegro, False se violado)
        e dicionário consolidado contendo métricas e detalhes das violações.
    """
    total_records = len(df)
    timestamp_str = datetime.now().isoformat()

    print(f"\n[AUDITORIA] Iniciando verificação de conformidade do lote: '{dataset_name}'...")
    print(f"[AUDITORIA] Volume amostral avaliado: {total_records} observações.")

    try:
        schema.validate(df, lazy=True)
        print(f"[STATUS] Lote '{dataset_name}' -> APROVADO sem nenhuma violação contratual.")
        report = {
            "dataset_name": dataset_name,
            "status": "APROVADO",
            "timestamp": timestamp_str,
            "total_records": total_records,
            "total_errors": 0,
            "error_categories": [],
            "failure_details": [],
        }
        return True, report

    except SchemaErrors as err:
        failure_cases = err.failure_cases
        total_errors = len(failure_cases)
        error_categories = failure_cases["check"].unique().tolist()

        print(f"[STATUS] Lote '{dataset_name}' -> REJEITADO com {total_errors} violações detectadas!")
        print("[BLOQUEIO] O pipeline interceptou as inconsistências e impediu a ingestão dos dados.")

        # Detalhamento resumido no console
        print("\n--- Sumário de Violações Interceptadas ---")
        summary_view = failure_cases[["column", "check", "failure_case"]].head(10)
        print(summary_view.to_string(index=False))
        if total_errors > 10:
            print(f"... e mais {total_errors - 10} ocorrências suprimidas na visualização.")

        report = {
            "dataset_name": dataset_name,
            "status": "REJEITADO",
            "timestamp": timestamp_str,
            "total_records": total_records,
            "total_errors": total_errors,
            "error_categories": error_categories,
            "failure_details": failure_cases.head(25).to_dict(orient="records"),
        }
        return False, report


# =============================================================================
# Injeção Controlada de Anomalias para Teste de Estresse (Falsificação Defensiva)
# =============================================================================

def create_corrupted_dataset_for_testing(
    df_clean: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Gera intencionalmente um lote de dados corrompido para testar a robustez do contrato.

    Injetam-se violações clássicas observadas em falhas de ingestão em produção:
        - Violação 1: Idade inferior à maioridade legal (age = 15) e valor negativo (age = -5).
        - Violação 2: Montantes de crédito zerados ou negativos (credit_amount = -1500).
        - Violação 3: Valores nulos em colunas mandatórias não anuláveis (checking_status = NaN).
        - Violação 4: Valores categóricos fora do domínio homologado (housing = 'spaceship').
        - Violação 5: Quebra de unicidade de chave primária (duplicação de client_id).
        - Violação 6: Violação de domínio de variável-alvo (target = 9).

    Parameters
    ----------
    df_clean : pd.DataFrame
        Dataset limpo de referência para sofrer adulterações controladas.
    random_state : int
        Semente pseudoaleatória.

    Returns
    -------
    pd.DataFrame
        Dataset adulterado contendo anomalias determinísticas.
    """
    df_bad = df_clean.copy()
    n = len(df_bad)

    print("\n[INJEÇÃO] Injetando anomalias deliberadas para teste de estresse do contrato:")

    # 1. Injeção de idade incompatível com maioridade legal (5 registros)
    df_bad.loc[0:2, "age"] = 15
    df_bad.loc[3:4, "age"] = -5
    print("  - Violação 1: Clientes menores de idade (15 anos) e valores etários negativos.")

    # 2. Injeção de valores de crédito negativos ou abusivos (5 registros)
    df_bad.loc[5:7, "credit_amount"] = -2500
    df_bad.loc[8:9, "credit_amount"] = 999999
    print("  - Violação 2: Montantes de crédito negativos (-2500) e acima do teto operacional.")

    # 3. Injeção de valores nulos em campos obrigatórios (5 registros)
    df_bad.loc[10:14, "checking_status"] = np.nan
    print("  - Violação 3: Valores nulos (NaN) introduzidos em campo obrigatório checking_status.")

    # 4. Injeção de categorias inválidas (5 registros)
    df_bad.loc[15:19, "housing"] = "spaceship_station"
    print("  - Violação 4: Categoria não homologada ('spaceship_station') atribuída a housing.")

    # 5. Duplicação de identificadores primários (5 registros)
    df_bad.loc[20:24, "client_id"] = df_bad.loc[0, "client_id"]
    print("  - Violação 5: Duplicidade forçada de identificador de cliente (client_id).")

    # 6. Variável-alvo fora do escopo binário (5 registros)
    df_bad.loc[25:29, "target"] = 9
    print("  - Violação 6: Alvo inconsistente (target = 9) fora do domínio binário {0, 1}.")

    return df_bad


# =============================================================================
# Execução Principal e Relatório Consolidado
# =============================================================================

def run_validation_pipeline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executa o pipeline completo da Etapa 1: validação limpa e teste com anomalias.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        Relatórios consolidados de validação do lote limpo e do lote corrompido.
    """
    ensure_reports_dir()

    ref_path = DATA_REFERENCE / "reference_data.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Arquivo de referência não localizado em '{ref_path}'. "
            "Execute previamente a Etapa 0 (python src/data_preparation.py)."
        )

    print("\n" + "=" * 70)
    print("ETAPA 1: VALIDAÇÃO DE DADOS E CONTRATOS FORMALIZADOS (PANDERA)")
    print("=" * 70)

    # 1. Carregamento dos dados de referência
    df_reference = pd.read_csv(ref_path)
    schema = build_credit_contract_schema()

    # 2. Validação do Dataset de Referência (Lote íntegro)
    valid_ref, report_ref = validate_dataset(df_reference, schema, "dataset_referencia_limpo")

    # 3. Geração do lote corrompido para validação do bloqueio
    df_corrupted = create_corrupted_dataset_for_testing(df_reference)

    # 4. Validação do lote corrompido (Lote adulterado)
    valid_corrupted, report_corrupted = validate_dataset(
        df_corrupted, schema, "dataset_lote_com_erros_intencionais"
    )

    # 5. Consolidação e exportação dos relatórios em JSON para governança
    consolidated_report = {
        "timestamp": datetime.now().isoformat(),
        "projeto": "Tech Challenge 04 - Credit Scoring",
        "dataset_avaliado": "German Credit Data (Kaggle/UCI)",
        "lote_referencia": report_ref,
        "lote_corrompido": report_corrupted,
        "conclusao": (
            "Contrato operando em conformidade estrita: dados limpos foram aprovados e "
            "lote corrompido foi integralmente interceptado e bloqueado."
            if (valid_ref and not valid_corrupted)
            else "Anomalia no mecanismo de validação."
        ),
    }

    report_file = REPORTS_DIR / "data_validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(consolidated_report, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCESSO] Relatório formal de auditoria gravado em: {report_file}")
    print("=" * 70 + "\n")

    return report_ref, report_corrupted


def main() -> None:
    """Ponto de entrada de linha de comando para a Etapa 1 (Validação)."""
    run_validation_pipeline()


if __name__ == "__main__":
    main()
