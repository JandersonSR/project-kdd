import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import io
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import arff
from arff import dump as arff_dump

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

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
        "Clusterização (K-Means)",
        "Avaliação dos Clusters",
        "Resumo Comparativo",
        "Resumo Comparativo e Exportação em PDF"
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

# ==============================
# VARIÁVEIS PERSISTENTES
# ==============================
if "df_scaled" not in st.session_state:
    st.session_state.df_scaled = None

if "numeric_continuous" not in st.session_state:
    st.session_state.numeric_continuous = None

if "kmeans_model" not in st.session_state:
    st.session_state.kmeans_model = None

if "X" not in st.session_state:
    st.session_state.X = None

if "k_final" not in st.session_state:
    st.session_state.k_final = None

if "cols_nominal" not in st.session_state:
    st.session_state.cols_nominal = None

if "numeric_cols" not in st.session_state:
    st.session_state.numeric_cols = None

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
elif menu == "Clusterização (K-Means)":
    st.header("🤖 Clusterização com K-Means")

    st.write("### 🔍 Carregando dataset…")
    st.write(df.head())

    # ============================================================
    # 2. Identificar atributos numéricos e transformar códigos em NOMINAL
    # ============================================================
    st.subheader("🧩 Identificação de colunas numéricas e nominais")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state.numeric_cols = numeric_cols

    cols_nominal = [col for col in numeric_cols if df[col].nunique() <= 10]

    st.write("**Colunas detectadas como NOMINAL:**", cols_nominal)

    df_nominal = df.copy()
    for col in cols_nominal:
        df_nominal[col] = df_nominal[col].astype("category")

    # ============================================================
    # 3. Preparar dados numéricos para normalização
    # ============================================================
    st.subheader("⚙ Normalização MinMax dos atributos contínuos")

    numeric_continuous = [c for c in numeric_cols if c not in cols_nominal]

    scaler = MinMaxScaler()
    df_scaled = df_nominal.copy()
    df_scaled[numeric_continuous] = scaler.fit_transform(df_nominal[numeric_continuous])

    st.write("**Colunas normalizadas:**", numeric_continuous)

    # ============================================================
    # 4. Gráfico do ELBOW
    # ============================================================
    st.subheader("📉 Método do Cotovelo (Elbow Method)")

    X = df_scaled[numeric_continuous].dropna()

    inertias = []
    K_range = range(2, 16)

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(K_range, inertias, marker='o')
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inércia")
    ax.set_title("Método do Cotovelo (Elbow Method)")
    ax.grid()
    st.pyplot(fig)

    st.info("📌 Escolha o valor ideal de K com base no gráfico acima.")

    # Campo para o usuário definir K
    k_final = st.number_input(
        "Escolha o número de clusters (k):",
        min_value=2,
        max_value=15,
        value=12,
        step=1
    )

    # ============================================================
    # 5. Executar K-Means
    # ============================================================
    st.subheader("🚀 Executando K-Means")

    if st.button("Rodar Clusterização"):
        
        kmeans = KMeans(n_clusters=k_final, random_state=42)
        df_scaled["cluster"] = kmeans.fit_predict(X)

        st.session_state.df_scaled = df_scaled
        st.session_state.numeric_continuous = numeric_continuous
        st.session_state.kmeans_model = kmeans
        st.session_state.X = X
        st.session_state.k_final = k_final
        st.session_state.cols_nominal = cols_nominal

        st.success(f"Clusterização concluída com k = {k_final} clusters!")
        st.write(df_scaled.head())

        # ============================================================
        # 6. Gerar ARFF (liac-arff)
        # ============================================================


        arff_data = df_scaled.copy()
        for col in cols_nominal:
            arff_data[col] = arff_data[col].astype(str)

        arff_dict = {
            "relation": "dataset_clusters",
            "attributes": [
                (col, "STRING") if col in cols_nominal else (col, "NUMERIC")
                for col in arff_data.columns
            ],
            "data": arff_data.values.tolist()
        }

        arff_buffer = StringIO()
        arff_dump(arff_dict, arff_buffer)

        # Conteúdo do arquivo como string
        arff_file = arff_buffer.getvalue()

        st.download_button(
            "📥 Baixar ARFF Clusterizado",
            arff_file,
            file_name="dataset_clusterizado.arff",
            mime="text/plain"
        )

        # ============================================================
        # 7. Salvar Excel com clusters
        # ============================================================
        df_final_excel = df.copy()
        df_final_excel["cluster"] = df_scaled["cluster"]

        excel_buffer = StringIO()
        df_final_excel.to_csv(excel_buffer, index=False)

        st.download_button(
            "📥 Baixar dataset final com clusters (CSV)",
            excel_buffer.getvalue(),
            file_name="dataset_com_clusters.csv",
            mime="text/csv"
        )

        st.success("Arquivos gerados com sucesso!")

elif menu == "Avaliação dos Clusters":
    st.header("📊 Avaliação dos Clusters (K-Means)")

    if st.session_state.df_scaled is None:
        st.error("⚠ Execute primeiro a opção 2 – Clusterização (K-Means).")
        st.stop()

    else:
        df_scaled = st.session_state.df_scaled
        numeric_continuous = st.session_state.numeric_continuous
        kmeans = st.session_state.kmeans_model
        X = st.session_state.X
        k_final = st.session_state.k_final
        cols_nominal = st.session_state.cols_nominal

        st.success("Clusters carregados com sucesso! ✔")

        # ============================================================
        # 1. Preparar dados
        # ============================================================
        X = df_scaled[numeric_continuous].dropna()
        labels = df_scaled["cluster"].values

        st.subheader("📈 Métricas de Avaliação")

        # ============================================================
        # 2. MÉTRICAS
        # ============================================================
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        st.write(f"**Silhouette Score:** `{sil_score:.4f}`")
        st.write(f"**Davies-Bouldin Index (menor melhor):** `{db_score:.4f}`")
        st.write(f"**Calinski-Harabasz (maior melhor):** `{ch_score:.2f}`")

        # Tamanho dos clusters
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        st.write("### 🔢 Tamanho dos clusters")
        st.write(cluster_sizes)

        # ============================================================
        # 3. CENTROIDES
        # ============================================================
        st.subheader("🧭 Centroides dos Clusters")
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_continuous)
        st.dataframe(centroids.round(4))

        # ============================================================
        # 4. SILHOUETTE PLOT
        # ============================================================
        st.subheader("📉 Silhouette Plot por Cluster")

        sample_silhouette_values = silhouette_samples(X, labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 10

        for i in unique:
            ith = sample_silhouette_values[labels == i]
            ith.sort()

            size_cluster_i = ith.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0, ith,
                alpha=0.7
            )
            ax.text(-0.05, (y_lower + y_upper) / 2, str(i))
            y_lower = y_upper + 10

        ax.axvline(x=sil_score, color="red", linestyle="--")
        ax.set_title("Silhouette Plot")
        ax.set_xlabel("Coeficiente de Silhouette")
        ax.set_ylabel("Amostras")

        st.pyplot(fig)

        # ============================================================
        # 5. PCA 2D
        # ============================================================
        st.subheader("🧭 Visualização PCA 2D dos Clusters")

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)

        df_pca = pd.DataFrame({
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "cluster": labels
        })

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_pca,
            x="PC1",
            y="PC2",
            hue="cluster",
            palette="tab10",
            ax=ax2
        )
        ax2.set_title("Clusters via PCA 2D")
        st.pyplot(fig2)

        # ============================================================
        # 6. HEATMAP
        # ============================================================
        st.subheader("🔥 Heatmap das Médias por Cluster")

        cluster_means = df_scaled.groupby("cluster")[numeric_continuous].mean()

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="viridis", ax=ax3)
        ax3.set_title("Médias dos atributos por cluster")
        st.pyplot(fig3)

elif menu == "Resumo Comparativo":
    st.header("📊 Resumo Comparativo Geral")

    if st.session_state.df_scaled is None:
        st.error("⚠ É necessário executar antes as opções 1, 2 e 3!")
        st.stop()

    else:
        df_scaled = st.session_state.df_scaled
        numeric_continuous = st.session_state.numeric_continuous
        kmeans = st.session_state.kmeans_model
        X = st.session_state.X
        k_final = st.session_state.k_final
        cols_nominal = st.session_state.cols_nominal
        
        st.success("Resumo consolidado de todas as etapas.")

        # ---------------------------------------------
        # SEÇÃO 1 — RESUMO DA OPÇÃO 1
        # ---------------------------------------------
        st.subheader("🟦 1. Estatísticas da Análise Exploratória (Opção 1)")

        st.write("**Número de atributos numéricos:**", len(st.session_state.numeric_cols))
        st.write("**Colunas consideradas NOMINAL (≤ 10 valores únicos):**")
        st.write(cols_nominal)

        # Assimetria média dos atributos
        skew_values = {col: df[col].skew() for col in st.session_state.numeric_cols}
        mean_skew = np.mean([abs(v) for v in skew_values.values()])

        st.write(f"**Assimetria média dos atributos:** `{mean_skew:.4f}`")


        # ---------------------------------------------
        # SEÇÃO 2 — RESUMO DA OPÇÃO 2
        # ---------------------------------------------
        st.subheader("🟩 2. Resultados da Clusterização (Opção 2)")

        st.write(f"**Número de clusters escolhidos (k):** `{k_final}`")

        # Tamanho dos clusters
        unique, counts = np.unique(df_scaled["cluster"].values, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        st.write("**Tamanho dos clusters:**")
        st.write(cluster_sizes)

        # Número de atributos normalizados
        st.write("**Atributos normalizados:**")
        st.write(numeric_continuous)


        # ---------------------------------------------
        # SEÇÃO 3 — RESUMO DA AVALIAÇÃO (Opção 3)
        # ---------------------------------------------
        st.subheader("🟧 3. Avaliação dos Clusters (Opção 3)")

        sil_score = silhouette_score(X, df_scaled["cluster"].values)
        db_score = davies_bouldin_score(X, df_scaled["cluster"].values)
        ch_score = calinski_harabasz_score(X, df_scaled["cluster"].values)

        st.write(f"**Silhouette Score:** `{sil_score:.4f}`")
        st.write(f"**Davies-Bouldin:** `{db_score:.4f}`  (menor melhor)")
        st.write(f"**Calinski–Harabasz:** `{ch_score:.2f}` (maior melhor)")

        # Melhor e pior cluster (por média de silhouette)
        from sklearn.metrics import silhouette_samples
        sil_samples = silhouette_samples(X, df_scaled["cluster"].values)

        cluster_mean_sil = {
            c: np.mean(sil_samples[df_scaled["cluster"] == c])
            for c in unique
        }

        best_cluster = max(cluster_mean_sil, key=cluster_mean_sil.get)
        worst_cluster = min(cluster_mean_sil, key=cluster_mean_sil.get)

        st.write(f"**Melhor cluster (silhouette médio):** `{best_cluster}`")
        st.write(f"**Pior cluster (silhouette médio):** `{worst_cluster}`")

        # ---------------------------------------------
        # SEÇÃO 4 — VISÃO FINAL
        # ---------------------------------------------
        st.subheader("🏁 Conclusão Geral")

        st.markdown("""
        ### 🔍 Insights Gerais:
        - A opção 1 confirmou quais atributos são realmente relevantes.
        - A opção 2 mostrou como os dados se agrupam sob normalização.
        - A opção 3 avaliou matematicamente a qualidade dos clusters.
        - A partir disso, você pode identificar padrões importantes, clusters dominantes e atributos críticos.

        ### 📤 Exportações:
        Você pode baixar os arquivos completos gerados na Opção 2 (ARFF e Excel).
        """)

elif menu == "Resumo Comparativo e Exportação em PDF":
    st.header("📊 Resumo Comparativo Geral + Exportação em PDF")

    if st.session_state.df_scaled is None:
        st.error("⚠ É necessário executar antes as opções 1, 2 e 3!")
        st.stop()
    else:
        df_scaled = st.session_state.df_scaled
        numeric_continuous = st.session_state.numeric_continuous
        kmeans = st.session_state.kmeans_model
        X = st.session_state.X
        k_final = st.session_state.k_final
        cols_nominal = st.session_state.cols_nominal

        st.success("Todas as etapas detectadas. Gerando resumo consolidado.")

        # ===========================
        # 1. Recalcular métricas
        # ===========================
        X = df_scaled[numeric_continuous].dropna()
        labels = df_scaled["cluster"].values

        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        sil_samples = silhouette_samples(X, labels)
        cluster_mean_sil = {c: np.mean(sil_samples[labels == c]) for c in unique}
        best_cluster = max(cluster_mean_sil, key=cluster_mean_sil.get)
        worst_cluster = min(cluster_mean_sil, key=cluster_mean_sil.get)

        # PCA para plot
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        df_pca = pd.DataFrame({
            "PC1": pca_result[:, 0],
            "PC2": pca_result[:, 1],
            "cluster": labels
        })

        # ===========================
        # 2. Mostrar resumo na tela
        # ===========================
        st.subheader("🔹 Métricas Gerais")
        st.write(f"**Silhouette Score:** `{sil_score:.4f}`")
        st.write(f"**Davies-Bouldin Index:** `{db_score:.4f}`")
        st.write(f"**Calinski-Harabasz Index:** `{ch_score:.2f}`")
        st.write("**Tamanho dos clusters:**", cluster_sizes)
        st.write(f"**Melhor cluster:** {best_cluster}")
        st.write(f"**Pior cluster:** {worst_cluster}")

        # ===========================
        # 3. Gerar figuras para o PDF
        # ===========================
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader

        # --- FIG 1: Silhouette Plot ---
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        y_lower = 10
        for i in unique:
            ith = sil_samples[labels == i]
            ith.sort()
            size_i = ith.shape[0]
            y_upper = y_lower + size_i
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, alpha=0.7)
            ax1.text(-0.05, (y_lower + y_upper) / 2, str(i))
            y_lower = y_upper + 10
        ax1.axvline(x=sil_score, color="red", linestyle="--")
        ax1.set_title("Silhouette Plot")
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png")
        buf1.seek(0)

        # --- FIG 2: PCA ---
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="cluster", palette="tab10", ax=ax2)
        ax2.set_title("Visualização PCA 2D")
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        buf2.seek(0)

        # --- FIG 3: Heatmap ---
        cluster_means = df_scaled.groupby("cluster")[numeric_continuous].mean()
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="viridis", ax=ax3)
        ax3.set_title("Heatmap das Médias dos Clusters")
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format="png")
        buf3.seek(0)

        # ===========================
        # 4. Criar PDF em memória
        # ===========================
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4
        margin = 40
        y = height - margin

        # TÍTULO
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin, y, "Resumo Comparativo do Projeto")
        y -= 40

        # TEXTO PRINCIPAL
        c.setFont("Helvetica", 12)
        text = f"""
Métricas Gerais:
----------------------------
Silhouette Score: {sil_score:.4f}
Davies-Bouldin Index: {db_score:.4f}
Calinski-Harabasz: {ch_score:.2f}

Tamanho dos Clusters: {cluster_sizes}

Melhor cluster (silhouette médio): {best_cluster}
Pior cluster (silhouette médio): {worst_cluster}
        """
        for line in text.split("\n"):
            c.drawString(margin, y, line)
            y -= 18

        # Função para adicionar imagens no PDF usando ImageReader
        def add_image(buf, y):
            img_height = 250
            img = ImageReader(buf)
            if y < img_height + margin:
                c.showPage()
                y = height - margin - img_height
            c.drawImage(img, margin, y, width=520, height=img_height)
            return y - img_height - 40

        y = add_image(buf1, y)
        y = add_image(buf2, y)
        y = add_image(buf3, y)

        c.save()
        pdf_buffer.seek(0)

        # ===========================
        # 5. Botão de download PDF
        # ===========================
        st.subheader("📥 Download do PDF Consolidado")
        st.download_button(
            label="📄 Baixar Relatório PDF",
            data=pdf_buffer,
            file_name="relatorio_comparativo.pdf",
            mime="application/pdf"
        )

        st.success("PDF gerado com sucesso!")
