import random
from torch.utils.data import Dataset

class SimpleTripletDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
        self.indices = list(range(len(sentences)))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        anchor = self.sentences[idx]
        pos = anchor  # positives Beispiel = identischer Satz

        # negatives Beispiel = zufälliger anderer Satz
        neg_idx = random.choice([i for i in self.indices if i != idx])
        neg = self.sentences[neg_idx]

        return anchor, pos, neg