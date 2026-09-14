# Análise de Dados para Machine Learning

**Aluno:** José Salustiano Neto Júnior
**Matrícula:** 20240062041
**Dataset:** [Social Media Addiction and Mental Wellbeing](https://www.kaggle.com/datasets/harpartapsingh13/social-media-addiction-and-mental-wellbeing-dataset/data)

## Descrição
Este repositório contém a análise exploratória e o pré-processamento de dados para a atividade da disciplina DCT1401 - Inteligência Artificial. O objetivo é preparar o dataset "Social Media Addiction and Mental Wellbeing" para a futura aplicação de modelos de Machine Learning focados em dois problemas principais: regressão e classificação.

## Estrutura
- `data/raw/`: Dataset original bruto.
- `data/processed/`: Datasets após limpeza, imputação e padronização.
- `notebooks/`: Notebooks com as análises detalhadas.
- `reports/figures/`: Gráficos gerados durante a EDA.
- `src/`: Funções auxiliares (se houver).

## Como Executar
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks na pasta `notebooks/` na seguinte ordem:
   - `01_eda.ipynb`
   - `02_preprocessing.ipynb`
   - `03_regression_analysis.ipynb`
   - `04_classification_analysis.ipynb`

## Principais Resultados
- Para **regressão**, a variável alvo escolhida foi `Mental_Wellbeing_Score` (justificativa detalhada no relatório em PDF, pautada em sua alta variabilidade e distribuição normal aproximada).
- Para **classificação**, a variável alvo definida foi `Addiction_Level` (justificativa detalhada no relatório em PDF, pois isola o comportamento de risco dos usuários).
- **Features mais relevantes identificadas:** `Daily_Usage_Hours`, `FOMO_Score`, `Scroll_Without_Purpose`, `Late_Night_Usage` e `Posts_Per_Week`.

## Contato
- Email: jose.salustiano.166@ufrn.edu.br