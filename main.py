"""
=============================================================================
Tech Challenge 04 — Pipeline Principal (Orquestrador)
=============================================================================

Módulo orquestrador responsável pela execução integrada e sequencial
das etapas do projeto de Observabilidade, Detecção de Drift e Governança:

    - Etapa 0: Ingestão, tratamento e preparação do German Credit Data.
    - Etapa 1: Validação de Contratos de Dados (Pandera) e Treino do Classificador Base (XGBoost).
    - Etapa 2: Simulação de Degradação Econômica (Data/Concept Drift) e Observabilidade (Evidently AI).

Uso:
    python main.py              -> Executa o ciclo completo (Etapas 0, 1 e 2)
    python main.py --etapa 0    -> Executa apenas ingestão e preparação
    python main.py --etapa 1    -> Executa validação de contratos e modelo base
    python main.py --etapa 2    -> Executa simulação de drift e relatórios Evidently
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Configuração de codificação para suporte universal em terminais Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_etapa_0() -> None:
    """Executa a Etapa 0: Ingestão e Preparação dos Dados."""
    from src.data_preparation import main as data_prep_main
    data_prep_main([])


def run_etapa_1() -> None:
    """Executa a Etapa 1: Validação de Dados (Data Contracts) e Modelo Baseline."""
    from src.data_validation import main as validation_main
    from src.model_training import main as training_main

    validation_main()
    training_main()


def run_etapa_2() -> None:
    """Executa a Etapa 2: Simulação e Detecção Estatística de Drift."""
    from src.drift_detection import main as drift_main
    drift_main()


def main() -> None:
    """
    Ponto de entrada do orquestrador do projeto.

    Processa argumentos de linha de comando e coordena a execução
    modular dos componentes de pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Tech Challenge 04 — Observabilidade Contínua e Detecção de Drift"
    )
    parser.add_argument(
        "--etapa",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Número da etapa específica a executar (0, 1 ou 2). Se omitido, executa o ciclo completo.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("  TECH CHALLENGE 04 — SUSTENTAÇÃO E OBSERVABILIDADE EM MACHINE LEARNING")
    print("  Instituição: Pós-Tech / FIAP | Machine Learning Engineering")
    print("  Dataset Oficial: German Credit Data (Kaggle/UCI - Statlog)")
    print("=" * 76)

    if args.etapa is None:
        print("\n>>> Executando pipeline completo (Etapas 0, 1 e 2)...")
        run_etapa_0()
        run_etapa_1()
        run_etapa_2()
    elif args.etapa == 0:
        print("\n>>> Executando especificamente a Etapa 0 (Preparação)...")
        run_etapa_0()
    elif args.etapa == 1:
        print("\n>>> Executando especificamente a Etapa 1 (Contratos e Modelo)...")
        run_etapa_1()
    elif args.etapa == 2:
        print("\n>>> Executando especificamente a Etapa 2 (Drift e Evidently AI)...")
        run_etapa_2()

    print("\n" + "=" * 76)
    print("  STATUS FINAL: Ciclo de execução finalizado com sucesso!")
    print("  Relatórios gerados disponíveis no diretório: ./reports/")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
