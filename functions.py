UNKNOWN_ID = 1
PAD_ID = 0

def build_vocab(texts):
    token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNKNOWN_ID}
    for text in texts:
        for tok in text.split():
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)
    return token_to_id, len(token_to_id)


def encode(text, token_to_id, max_len=None):
    tokens = text.split()
    ids = [token_to_id.get(t, UNKNOWN_ID) for t in tokens]
    if max_len is not None:
        ids = ids[:max_len]
        ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids