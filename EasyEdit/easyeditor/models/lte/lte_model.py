import torch
import numpy as np

from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationConfig
from transformers.utils import ModelOutput
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer, util
from typing import Optional
from ...util.hparams import HyperParams

class LTEModel(PreTrainedModel):
    def __init__(
            self,
            origin_model,
            tokenizer,
            lora_model,
            sentence_model,
            hparams
        ):
        super().__init__(origin_model.config)
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.origin_model = origin_model
        self.lora_model = lora_model
        self.sentence_model = sentence_model
        self.device_name = f'cuda:{hparams.device}' if torch.cuda.is_available() and hparams.device >= 0 else 'cpu'

        self.k = hparams.k
        self.embeddings = np.array([])
        self.facts = []

        # self.origin_model.to(self.device_name)
        # if self.lora_model:
        #     self.lora_model.to(self.device_name)
        # if self.sentence_model:
        #     self.sentence_model.to(self.device_name)

    @classmethod
    def from_pretrained(cls, 
                      base_model_path: str, 
                      lora_model_path: Optional[str] = None,
                      sentence_model_path: Optional[str] = None,
                      **kwargs):
        origin_model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        
        lora_model = None
        if lora_model_path:
            lora_model = PeftModel.from_pretrained(origin_model, lora_model_path)

        sentence_model = None
        if sentence_model_path:
            sentence_model = SentenceTransformer(sentence_model_path)
        
        return cls(origin_model, lora_model, sentence_model, kwargs)
    
    def edit(self, query, new_fact, keep_original_weight=False):
        query_vector = self.sentence_model.encode(query, show_progress_bar=False)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        self.embeddings = np.concatenate((self.embeddings, query_vector), axis=0) if self.embeddings.size else query_vector
        self.facts.append(new_fact)

        if keep_original_weight:
            self.embeddings = self.embeddings[-1:]
            self.facts = self.facts[-1:]
        
    def sentence_retrieval(
        self,
        query_sentence: str,
    ):
        try:
            stored_embeddings = torch.tensor(self.embeddings).detach().cpu()
            stored_embeddings = util.normalize_embeddings(stored_embeddings)

            query_embedding = util.normalize_embeddings(torch.tensor(self.sentence_model.encode(
                query_sentence, show_progress_bar=False)).unsqueeze(0).detach().cpu())

            hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=self.k)
            assert len(hits) == 1
            hit = hits[0]
            retrieved_sentences = [self.facts[hit[k]["corpus_id"]] for k in range(len(hit))]
            
            return retrieved_sentences
        except Exception as e:
            print(f"Error in sentence retrieval: {e}")
            return ''
    
    def retrieve_prefix(
        self,
        input_text: str,
    ):
        retrieved_facts = self.sentence_retrieval(input_text)

        if len(retrieved_facts) > 1:
            processed_text = "Please acknowledge the updated information provided below and respond to the subsequent question.\n\n[Updated Information]:"
            for i in range(len(retrieved_facts)):
                processed_text += f"\n{i+1}. " + retrieved_facts[i]
            processed_text += "\n\n[Question]:\n"
        else:
            processed_text = "Please acknowledge the updated information provided below and respond to the subsequent question.\n\n[Updated Information]:\n" \
                + retrieved_facts[0] + "\n\n[Question]:\n"
                
        return processed_text

    def forward(
            self,
            pre_edit: bool = False,
            **kwargs
        ):
        if pre_edit:
            with self.lora_model.disable_adapter():
                output = self.origin_model(**kwargs)
        else:
            output = self.lora_model(**kwargs)
        return output

    def generate(
            self,
            pre_edit: bool = False,
            **kwargs
        ):
        if pre_edit:
            with self.lora_model.disable_adapter():
                output = self.origin_model.generate(
                    **kwargs,
                    generation_config=GenerationConfig(
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    ),
                )
        else:
            output = self.lora_model.generate(
                **kwargs,
                generation_config=GenerationConfig(
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                ),
            )
        return output
