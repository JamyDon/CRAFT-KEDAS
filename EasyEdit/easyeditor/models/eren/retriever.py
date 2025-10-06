from typing import List
import torch
from torch import Tensor
from .embedder import Embedder

class Retriever:
    """
    Interface for a retriever
    """
    def add_to_index(self, input_texts: List[str]):
        raise NotImplementedError

    def retrieve(self, input_texts: str, topk: int):
        raise NotImplementedError

class MipsRetriever(Retriever):
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.embeds_cache = None

    def clear(self):
        self.embeds_cache = None

    def add_to_index(self, sents: List[str]):
        """Embed sentences and add them to index."""
        embeds = self.embedder.embed(sents)  # (n, d)
        if self.embeds_cache is None:
            self.embeds_cache = embeds
        else:
            self.embeds_cache = torch.cat([self.embeds_cache, embeds], dim=0)

    def retrieve(self, input_texts: str, topk: int) -> Tensor:
        """
        Return top-k (similarity scores, index).
        n: number of inputs.
        d: embedding dim.
        c: cache size (number of embeddings in the index)
        k: return top-k indices
        """
        cache_size, embed_dim = self.embeds_cache.shape
        ctx_embeds = self.embeds_cache.view(cache_size, embed_dim)  # (c, d)
        input_embeds = self.embedder.embed(input_texts)  # (n, d)
        sim_scores = input_embeds @ ctx_embeds.T  # (n, c)
        sims, idxs = torch.topk(sim_scores, min(topk, cache_size), dim=1)
        return sims, idxs 