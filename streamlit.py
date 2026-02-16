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


st.sidebar.title("Rafiki Augmented Generation")
question = st.sidebar.text_area("Frage eingeben", "Schneller Schoko‑Becherkuchen in der Tasse mit Mehl, Kakao, Zucker, Backpulver, und Schokostückchen für die Mikrowelle", height="content")
embedding_dimensions = st.sidebar.slider("Dimensions", 3, 100, 3)
epochs = st.sidebar.slider("Epochen", 1, 1000, 500)

tab1, tab2, tab3, tab4 = st.tabs(["🧠 Wissensschatz", "📈 Darstellung", "📋 Embeddings", "↔ Vergleich"])

with tab1:
    doc1 = st.text_area("Dokument 1", "Cremige Zitronen‑Tagliatelle: Koche Tagliatelle al dente und reibe währenddessen die Schale einer Bio‑Zitrone fein ab. In einer Pfanne erhitzt du etwas Butter, gibst Sahne, Zitronenschale, Zitronensaft, Salz und Pfeffer dazu und lässt alles leicht einkochen. Die Nudeln direkt aus dem Kochwasser in die Sauce geben, gut vermengen und mit frisch geriebenem Parmesan und etwas schwarzem Pfeffer servieren.", height="content")
    doc2 = st.text_area("Dokument 2", "Knusprige Hähnchenpfanne mit Paprika: Schneide Hähnchenbrust in Streifen und brate sie in Olivenöl kräftig an. Füge rote und gelbe Paprika, Zwiebelstreifen und Knoblauch hinzu und brate alles weiter, bis das Gemüse weich wird. Mit Paprikapulver, Salz, Pfeffer und einem Schuss Sojasauce würzen. Zum Schluss etwas Petersilie unterheben und mit Reis oder Brot genießen.", height="content")
    doc3 = st.text_area("Dokument 3", "Mediterraner Couscous‑Salat: Bereite Couscous mit heißer Gemüsebrühe zu und lasse ihn locker auskühlen. Mische gewürfelte Tomaten, Gurke, rote Zwiebel, Oliven und Feta unter. Für das Dressing verrührst du Olivenöl, Zitronensaft, Salz, Pfeffer und etwas Honig. Alles gut vermengen, kurz ziehen lassen und mit frischer Minze servieren.", height="content")
    doc4 = st.text_area("Dokument 4", "Schneller Schoko‑Becherkuchen: In einer Tasse mischst du Mehl, Kakao, Zucker, Backpulver und eine Prise Salz. Gib Milch, etwas Öl und ein paar Schokostückchen dazu und rühre zu einem glatten Teig. Die Tasse für etwa 60–90 Sekunden in die Mikrowelle stellen, bis der Kuchen aufgegangen ist. Kurz abkühlen lassen und warm genießen.", height="content")

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