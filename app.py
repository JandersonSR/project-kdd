import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import textwrap
from io import StringIO

# ====================================================
# CONFIGURAÇÃO INICIAL
# ====================================================
st.set_page_config(page_title="Projeto de Análise", layout="wide")
st.title("📊 Projeto de Análise de Dados com Streamlit")

# ====================================================
# MENU LATERAL
# ====================================================
menu = st.sidebar.radio(
    "Selecione uma opção:",
    [
        "Gerar Histogramas", 
        "Análise Estatística (placeholder)",
        "Modelos (placeholder)"
    ]
)

# ====================================================
# FUNÇÃO PARA CARREGAR DADOS
# ====================================================
@st.cache_data
def load_data():
    path = "https://docs.google.com/spreadsheets/d/1supejFq9cpVVY_doGhtny902ti7H7rcTaaMvsFFGI3M/export?format=csv"
    return pd.read_csv(path, low_memory=False)

df = load_data()


# ====================================================
# OPÇÃO 1 – GERAR HISTOGRAMAS
# ====================================================
if menu == "Gerar Histogramas":

    st.header("📈 Gerador de Histogramas e Relatório Automático")

    st.subheader("📁 Dataset Carregado")
    st.write(df.head())

    # Remover colunas sem repetição
    unique_columns = [col for col in df.columns if df[col].nunique() == len(df)]
    df_clean = df.drop(columns=unique_columns, errors="ignore")

    st.subheader("🚮 Colunas Removidas (valores totalmente únicos)")
    st.write(unique_columns if unique_columns else "Nenhuma coluna removida.")

    # Identificar colunas numéricas
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("🔢 Colunas Numéricas Detectadas")
    st.write(numeric_cols)

    # Histogramas
    st.subheader("📊 Histogramas")
    for col in numeric_cols:
        st.write(f"### Histograma – {col}")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_clean[col].dropna(), bins=30, color='steelblue', edgecolor='black')
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Gerar relatório
    report = []
    report.append("RELATÓRIO DA DISTRIBUIÇÃO DOS DADOS\n")
    report.append("=" * 60 + "\n")

    for col in numeric_cols:
        desc = df_clean[col].describe()
        skew = df_clean[col].skew()
        unique_vals = df_clean[col].nunique()

        text = f"""
Atributo: {col}
-----------------------------------------
Mínimo: {desc['min']}
Máximo: {desc['max']}
Média: {desc['mean']}
Desvio Padrão: {desc['std']}
Assimetria (skew): {skew:.4f}
Valores distintos: {unique_vals}

Interpretação:
- {'Assimetria forte → possível discretização necessária.' if abs(skew) > 1 else 'Assimetria moderada ou leve.'}
- {'Poucos valores distintos → Pode ser categórico (numeric-to-nominal).' if unique_vals <= 10 else 'Atributo contínuo → Mantido como numérico.'}
"""
        report.append(text)

    # Colunas que devem virar nominal
    cols_nominal = [col for col in numeric_cols if df_clean[col].nunique() <= 10]
    report.append("\n=== COLUNAS QUE DEVEM VIRAR NOMINAL ===")
    report.append(str(cols_nominal))

    # Mostrar relatório
    st.subheader("📄 Relatório Gerado")
    full_report = "\n".join(report)
    st.text(full_report)

    # Download
    st.download_button(
        label="📥 Baixar Relatório TXT",
        data=full_report,
        file_name="relatorio_etapa1.txt",
        mime="text/plain"
    )


# ====================================================
# OPÇÃO 2 – PLACEHOLDER
# ====================================================
elif menu == "Análise Estatística (placeholder)":
    st.header("📘 Análise Estatística")
    st.info("Esta opção será adicionada em breve.")


# ====================================================
# OPÇÃO 3 – PLACEHOLDER
# ====================================================
elif menu == "Modelos (placeholder)":
    st.header("🤖 Modelos de Machine Learning")
    st.info("Esta opção será implementada posteriormente.")
