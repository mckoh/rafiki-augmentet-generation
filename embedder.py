# 3. Kleines Embedding-Modell (3D)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L


class ParagraphEmbedder(nn.Module):
    def __init__(self, vocab_size, emb_dim=3, pad_idx=0):
        super().__init__()

        self.pad_id = pad_idx

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_idx
        )

    def forward(self, x):
        """
        x: LongTensor (batch, seq_len)
        Rückgabe: (batch, emb_dim) – Mittelwert über die Sequenz
        """
        emb = self.embedding(x)          # (batch, seq_len, emb_dim)
        mask = (x != self.pad_id).unsqueeze(-1)  # (batch, seq_len, 1)
        emb = emb * mask                 # PADs auf 0 setzen
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        sent_emb = emb.sum(dim=1) / lengths    # (batch, emb_dim)
        return sent_emb

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        pos = self.data[(idx + 1) % len(self.data)]
        neg = self.data[(idx + 2) % len(self.data)]
        return anchor, pos, neg


class LightningParagraphEmbedder(L.LightningModule):
    def __init__(self, vocab_size, pad_id, emb_dim=3, lr=1e-3):
        super().__init__()
        self.model = ParagraphEmbedder(vocab_size)
        self.pad_id = pad_id
        self.lr = lr
        self.loss = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        anchor, pos, neg = batch

        emb_a = self(anchor)
        emb_p = self(pos)
        emb_n = self(neg)

        loss = F.triplet_margin_loss(emb_a, emb_p, emb_n, margin=1.0)
        self.loss.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)