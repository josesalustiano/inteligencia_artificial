# Inteligência Artificial - Análise de Dados e Avaliação Comparativa de Modelos (UFRN)

**Aluno:** José Salustiano Neto Júnior  
**Matrícula:** 20240062041  
**Disciplina:** DCT1401 - Inteligência Artificial  
**Professor:** Thommas K. S. Flores  
**Dataset:** [Social Media Addiction and Mental Wellbeing](https://www.kaggle.com/datasets/harpartapsingh13/social-media-addiction-and-mental-wellbeing-dataset/data)

---

## 📌 Descrição do Projeto

Este repositório centraliza o desenvolvimento prático da disciplina de Inteligência Artificial, dividido em duas etapas complementares utilizando o dataset *"Social Media Addiction and Mental Wellbeing"*:

* **Etapa 1 — Análise Exploratória e Pré-processamento:** Diagnóstico da qualidade dos dados brutos, imputação de nulos, tratamento de outliers, padronização (Z-score) e seleção dos alvos de Regressão (`Mental_Wellbeing_Score`) e Classificação (`Addiction_Level`).
* **Etapa 2 — Avaliação Comparativa de Modelos:** Construção, benchmark e avaliação comparativa entre três famílias de modelos:
  1. **Sistema Especialista:** Base de regras SE-ENTÃO com motor de inferência próprio e medição do esforço humano de elaboração.
  2. **Árvore de Decisão:** Modelos podados e sem restrição (para evidenciar overfitting) para regressão e classificação.
  3. **Ensemble Learning:** Random Forest (com análise de viés/variância) e Gradient Boosting.

---

## 📂 Estrutura do Repositório

```text
inteligencia_artificial/
├── data/
│   ├── raw/                 # Dataset original bruto inviolado
│   └── processed/           # Conjuntos limpos e divididos em treino/teste (estratificado)
├── docs/
│   └── relatorio_final.pdf  # Relatório técnico em PDF
├── logs/
│   ├── cronometro_regras.csv # Registro temporal das sessões de elaboração do Sistema Especialista
│   └── emissions.csv         # Logs de consumo de energia e emissão de CO2 (CodeCarbon)
├── models/
│   └── rf_regression_model.pkl # Artefatos binários dos modelos treinados
├── notebooks/
│   ├── 01_eda.ipynb                     # [Etapa 1] Análise exploratória inicial
│   ├── 02_preprocessing.ipynb           # [Etapa 1] Limpeza, imputação e escalonamento
│   ├── 03_regression_analysis.ipynb     # [Etapa 1] Percepções do alvo contínuo
│   ├── 04_classification_analysis.ipynb # [Etapa 1] Percepções do alvo categórico
│   ├── 08_regression_models.ipynb       # [Etapa 2] Ajuste de hiperparâmetros e treino
│   ├── 09_generalization.ipynb          # [Etapa 2] Gap de generalização e K-Fold CV
│   ├── 10_efficiency.ipynb              # [Etapa 2] Profiling de tempo, RAM e CO2
│   ├── 11_model_visualization.ipynb    # [Etapa 2] Curvas ROC, PR-AUC, resíduos e árvores
│   ├── 12_final_metrics.ipynb           # [Etapa 2] Consolidação das métricas exatas
│   └── 13_table.ipynb                   # [Etapa 2] Geração automatizada da tabela-resumo
├── reports/
│   ├── figures/             # Gráficos (matriz de confusão, resíduos, ROC/PR, árvores)
│   └── tables/              # Tabelas consolidadas em CSV
├── rules/
│   ├── regras_classificacao.txt # Base de regras SE-ENTÃO (Classificação)
│   └── regras_regressao.txt     # Base de regras SE-ENTÃO (Regressão)
├── src/
│   ├── data_split.py        # Divisão reprodutível treino/teste
│   ├── expert_system.py     # Base de regras e motor de inferência
│   └── profiling.py         # Medição de tempo, memória e recursos
├── README.md                # Documentação e guia de navegação
└── requirements.txt         # Mapeamento de dependências

```

--- 

### 2. Execução em Ordem dos Notebooks

Para reproduzir integralmente as duas etapas do trabalho, execute os notebooks na pasta `notebooks/` na seguinte ordem:

* **Etapa 1:** `01_eda.ipynb` ➔ `02_preprocessing.ipynb` ➔ `03_regression_analysis.ipynb` ➔ `04_classification_analysis.ipynb`
* **Etapa 2:** `05_decision_tree.ipynb` ➔ `06_ensemble_learning.ipynb` ➔ `07_classification_plots.ipynb` ➔ `08_regression_models.ipynb` ➔ `09_generalization.ipynb` ➔ `10_efficiency.ipynb` ➔ `11_model_visualization.ipynb` ➔ `12_final_metrics.ipynb` ➔ `13_table.ipynb`


## 📊 Principais Resultados

### Etapa 1 — Análise Exploratória e Alvos

* **Regressão:** Alvo `Mental_Wellbeing_Score` (alta variabilidade e distribuição normal aproximada).
* **Classificação:** Alvo `Addiction_Level` (isolamento do comportamento e grupos de risco).
* **Features Principais:** `Daily_Usage_Hours`, `FOMO_Score`, `Scroll_Without_Purpose`, `Late_Night_Usage` e `Posts_Per_Week`.

### Etapa 2 — Comparativo entre Modelos

* **Desempenho Preditivo:** O **Random Forest** obteve o melhor desempenho no teste em ambos os problemas (Regressão: RMSE = 0.475; Classificação: F1-Score = 0.885, ROC-AUC = 0.936), reduzindo a variância das árvores isoladas.
* **Generalização:** O **Random Forest** apresentou o menor gap de generalização (0.019), enquanto a **Árvore de Decisão** sem restrições sofreu severo *overfitting* (gap de 0.279).
* **Eficiência Computacional:** A **Árvore de Decisão Podada** provou ser a mais viável para cenários embarcados/edge (inferência em 0.0027 ms/amostra, alocação de apenas 132.8 KB de RAM e consumo energético com emissão irrisória de CO₂).
* **Esforço Humano vs. Treino:** O **Sistema Especialista** exigiu 95 minutos de esforço humano contínuo (cronometrado em 6 sessões) para codificar 21 regras, enquanto os modelos de Machine Learning foram treinados computacionalmente em frações de segundo (0.11s a 0.58s).

## Como Executar
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks na pasta `notebooks/` na seguinte ordem:
   - `01_eda.ipynb`
   - `02_preprocessing.ipynb`
   - `03_regression_analysis.ipynb`
   - `04_classification_analysis.ipynb`
