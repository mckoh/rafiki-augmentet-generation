import streamlit as st
import torch
import torch.nn as nn
import lightning as L

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
question = st.sidebar.text_area("Frage eingeben", "Schneller Schoko‑Becherkuchen in der Tasse mit Mehl, Kakao, Zucker, Backpulver, und Schokostückchen für die Mikrowelle", height="content")
embedding_dimensions = st.sidebar.slider("Dimensions", 3, 100, 3)
epochs = st.sidebar.slider("Epochen", 1, 1000, 500)

tab1, tab2, tab3, tab4 = st.tabs(["🧠 Wissensschatz", "📈 Darstellung", "📋 Embeddings", "↔ Vergleich"])

with tab1:
    doc1 = st.text_area("Dokument 1", "Simba ist der junge Löwenprinz und die zentrale Figur in Der König der Löwen. Als Sohn von Mufasa wächst Simba mit dem Gefühl auf, eines Tages König des Geweihten Landes zu werden. Nach dem Verlust seines Vaters flieht er, findet jedoch mit Timon und Pumbaa neue Freunde. Simbas Entwicklung vom sorglosen Jungtier zum verantwortungsvollen Herrscher bildet den Kern der Geschichte.", height="content")
    doc2 = st.text_area("Dokument 2", "Mufasa ist der weise und gerechte König des Geweihten Landes. Er verfügt über eine starke, majestätische Präsenz und lehrt Simba wichtige Werte wie Verantwortung und Mut. Seine berühmte Vorstellung vom „Kreis des Lebens“ prägt die Welt von Der König der Löwen. Obwohl Mufasa früh stirbt, bleibt er durch Simbas Erinnerungen und seine spirituelle Erscheinung eine Leitfigur.", height="content")
    doc3 = st.text_area("Dokument 3", "Scar ist Mufasas Bruder und der Hauptantagonist der Geschichte. Er ist intelligent, manipulativ und verfolgt skrupellos das Ziel, selbst König zu werden. Durch eine Intrige, in deren Folge Mufasa stirbt und Simba flieht, reißt Scar die Herrschaft an sich. Unter seiner Regentschaft verfällt das Geweihte Land, da er die natürlichen Regeln des Lebens ignoriert.", height="content")
    doc4 = st.text_area("Dokument 4", "Timon, ein Erdmännchen, und Pumbaa, ein Warzenschwein, sind ein humorvolles Duo, das Simba nach seiner Flucht aufnimmt. Sie leben nach dem Motto „Hakuna Matata“, das sie Simba beibringen. Dieses Lebensprinzip hilft Simba zunächst, seine Sorgen zu vergessen, stellt aber später ein Hindernis dar, als er sich seiner Verantwortung stellen muss. Timon und Pumbaa bleiben Simbas treue Freunde und unterstützen ihn im finalen Kampf.", height="content")

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
    # st.session_state["model"] = ParagraphEmbedder(vocab_size=vocab_size, emb_dim=embedding_dimensions)

if st.sidebar.button("Train"):
    train()
    # st.session_state["model"] = ParagraphEmbedder(vocab_size=vocab_size, emb_dim=embedding_dimensions)

# ==================== Model application ====================
with torch.no_grad():
    document_embeddings = st.session_state["model"](encoded_documents)
    question_embeddings = st.session_state["model"](encoded_question)

# ==================== Visualization ====================
with tab2:
    points = document_embeddings.numpy()
    qpoints = question_embeddings.numpy()
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=80, c='royalblue')
    ax.scatter(qpoints[:, 0], qpoints[:, 1], qpoints[:, 2], s=80, c='red')
    for i, point in enumerate(points):
        ax.text(point[0], point[1], point[2], f"D{i+1}", fontsize=10)
    ax.text(qpoints[0, 0], qpoints[0, 1], qpoints[0, 2], f"Q", fontsize=10)
    st.pyplot(fig)

with tab3:
    st.header("Embeddings der Dokumente")
    st.write(document_embeddings)
    st.header("Embeddings der Frage")
    st.write(question_embeddings)

with tab4:
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    all_embeddings = concat([question_embeddings, document_embeddings])
    st.write(all_embeddings)
    Z = hierarchy.linkage(all_embeddings, method="ward")
    hierarchy.dendrogram(Z, orientation='left', ax=ax, labels=["Frage"] + [f"D{i}" for i in range(1, len(all_embeddings))])
    st.pyplot(fig)


fig, ax = plt.subplots()
ax.plot(st.session_state["model"].loss)
st.sidebar.pyplot(fig)