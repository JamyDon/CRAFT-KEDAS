from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch import nn
from typing import List, Iterable
from ...util.hparams import HyperParams
from .embedder import Contriever
from .retriever import MipsRetriever
from .utils import gen_texts_mrc

class ERENModel(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        hparams: HyperParams
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.num_context_examples = hparams.num_context_examples
        self.hparams = hparams
        self.notes: List[str] = []
        self.name_or_path = self.model.name_or_path
        # Initialize embedder and retriever internally
        self.embedder = Contriever()
        self.retriever = MipsRetriever(self.embedder)
        self.device_name = f"cuda:{hparams.device}"
        self.model.to(self.device_name)

    def clear(self):
        self.notes = []
        self.retriever.clear()

    def edit(self, statements: List[str], keep_original_weight=False):
        if keep_original_weight:
            self.clear()

        processed = [s[:-1] if s.endswith('.') else s for s in statements]
        self.notes.extend(processed)
        self.retriever.add_to_index(processed)

    def create_context(self, context_idxs: Iterable) -> str:
        context_statements = [self.notes[i] for i in context_idxs]
        context = ".\n".join(context_statements) + "."
        return context
    
    def retrieve_prefix(self, question: str) -> str:
        _, ctx_idxs = self.retriever.retrieve(
            [question], topk=self.num_context_examples
        )
        contexts = [self.create_context(idxs) for idxs in ctx_idxs]
        output_text = gen_texts_mrc(
            self.model,
            self.tokenizer,
            contexts,
            [question],
            max_length=10,
        )[0]
        if output_text.strip().lower() == "unanswerable":
            return "Please answer this question: "
        else:
            return "Read this and answer the question.\n\n{context[0]}\n\n"
    
    def forward(
            self,
            **kwargs
        ):
        return self.model(**kwargs)

    def generate(
            self,
            **kwargs
        ):
        return self.model.generate(
            **kwargs,
            generation_config=GenerationConfig(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )