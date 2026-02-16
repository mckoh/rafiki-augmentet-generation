import streamlit as st
import torch
import torch.nn as nn
import lightning as L
from pandas import DataFrame

from functions import build_vocab, encode, PAD_ID
from embedder import ParagraphEmbedder, TripletDataset, LightningParagraphEmbedder
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
from numpy import concat
from torch.utils.data import DataLoader


st.set_page_config(
    page_title="Rafiki Augmented Generation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Rafiki Augmented Generation")

question = st.sidebar.text_area("Frage eingeben", "Der junge Löwenprinz der vom sorglosen Jungtier zum Herrscher wird.", height="content")
embedding_dimensions = st.sidebar.slider("Dimensions", 3, 100, 3)
epochs = st.sidebar.slider("Epochen", 1, 700, 250)

col1, col2 = st.columns([0.3, 0.7], gap="medium")

with col1:
    st.header("Dokumente")
    doc1 = st.text_area("Dokument 1", "Simba ist der junge Löwenprinz der vom sorglosen Jungtier zum verantwortungsvollen Herrscher wird.", height="content")
    doc2 = st.text_area("Dokument 2", "Mufasa ist der König des Geweihten Landes. Er verfügt über eine starke, majestätische Präsenz.", height="content")
    doc3 = st.text_area("Dokument 3", "Scar ist der böse Onkel, der selbst zum König werden will.", height="content")
    doc4 = st.text_area("Dokument 4", "Timon ist ein Erdmännchen und wird durch den Spruch Hakuna Matata berühmt.", height="content")

with col2:
    tab2, tab3, tab4 = st.tabs(["📋 Embeddings", "📈 Darstellung", "↔ Vergleich"])

    # ==================== Model Training ====================
    documents = [doc1, doc2, doc3, doc4]
    vocab, vocab_size = build_vocab(documents)
    max_len = max(len(document.split()) for document in documents)

    encoded_documents = [encode(document, vocab, max_len=max_len) for document in documents]
    encoded_documents = torch.tensor(encoded_documents)

    encoded_question = [encode(question, vocab, max_len=max_len)]
    encoded_question = torch.tensor(encoded_question)


    def train():
        dataset = TripletDataset(encoded_documents)
        loader = DataLoader(dataset, batch_size=vocab_size, shuffle=True)
        st.session_state["model"] = LightningParagraphEmbedder(
            vocab_size=vocab_size,
            pad_id=PAD_ID,
            emb_dim=embedding_dimensions,
            lr=0.01
        )

        st.session_state["trainer"] = L.Trainer(max_epochs=epochs, enable_checkpointing=False, logger=False)
        st.session_state["trainer"].fit(st.session_state["model"], loader)


    if "model" not in st.session_state:
        train()

    if st.sidebar.button("Train"):
        train()

    # ==================== Model application ====================
    with torch.no_grad():
        document_embeddings = st.session_state["model"](encoded_documents).numpy()
        question_embeddings = st.session_state["model"](encoded_question).numpy()

    # ==================== Visualization ====================
    with tab3:
        points = document_embeddings
        qpoints = question_embeddings
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=80, c='royalblue')
        ax.scatter(qpoints[:, 0], qpoints[:, 1], qpoints[:, 2], s=80, c='red')
        for i, point in enumerate(points):
            ax.text(point[0], point[1], point[2], f"D{i+1}", fontsize=10)
        ax.text(qpoints[0, 0], qpoints[0, 1], qpoints[0, 2], f"Q", fontsize=10)
        st.pyplot(fig)

    with tab2:
        cols = [f"Dimension {i+1}" for i in range(len(document_embeddings[0]))]
        st.header("Embeddings der Dokumente")
        df1 = DataFrame(document_embeddings)
        df1.columns = cols
        st.write(df1)

        st.header("Embeddings der Frage")
        df2 = DataFrame(question_embeddings)
        df2.columns = cols
        st.write(df2)

    with tab4:
        fig, ax = plt.subplots()
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        all_embeddings = concat([question_embeddings, document_embeddings])
        Z = hierarchy.linkage(all_embeddings, method="ward")
        hierarchy.dendrogram(Z, orientation='left', ax=ax, labels=["Frage"] + [f"D{i}" for i in range(1, len(all_embeddings))])
        st.pyplot(fig)


fig, ax = plt.subplots()
ax.plot(st.session_state["model"].loss)
st.sidebar.pyplot(fig)