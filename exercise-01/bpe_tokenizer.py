from helper import get_stats, merge


class BpeTokenizer:

    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.merges = {}

    def train(self, text, vocab_size):
        """Train the vocabulary using the Byte Pair Encoding (BPE) algorithm."""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # iteratively merge the most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)

            # find the pair with the highest count
            pair = max(stats, key=stats.get)

            # mint a new token: assign it the next available id
            idx = 256 + i

            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        """Given ids (list of integers), return Python string."""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """Given a string text, return the token ids."""
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:

            # find the pair with the lowest merge index
            stats = get_stats(ids)

            # if there are no more merges available, the key will result in an infinite for every single pair,
            # and the min will be the first pair in the list, arbitrarily
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
