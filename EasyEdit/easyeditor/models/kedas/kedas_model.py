import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationConfig
from transformers.utils import ModelOutput
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer, util
from typing import Optional

from .knowledge_base import KnowledgeBase

class KEDASModel(PreTrainedModel):
    def __init__(
            self,
            origin_model,
            tokenizer,
            lora_model,
            # sentence_model,
            hparams
        ):
        super().__init__(origin_model.config)
        self.tokenizer = tokenizer
        self.origin_model = origin_model
        self.lora_model = lora_model
        self.knowledge_base = KnowledgeBase(hparams)
        self.device_name = f'cuda:{hparams.device}' if torch.cuda.is_available() and hparams.device >= 0 else 'cpu'

        self.self_adaptive_post_alignment_inference = hparams.self_adaptive_post_alignment_inference
        self.alignment = hparams.alignment

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
    
    def edit(self, request, target_new, keep_original_weight=False):
        if keep_original_weight:
            # del self.knowledge_base
            # self.knowledge_base = KnowledgeBase(self.hparams)
            self.knowledge_base.clear()
        self.knowledge_base.add(request, target_new)

    def retrieve_prefix(
            self,
            input_text: str,
        ):
        retrieved_facts = self.knowledge_base.search(input_text)
        if len(retrieved_facts) > 1:
            processed_texts = "Please acknowledge the updated information provided below and respond to the subsequent question.\n\n[Updated Information]:"
            for i in range(len(retrieved_facts)):
                processed_texts += f"\n{i+1}. " + retrieved_facts[i]
            processed_texts += "\n\n[Question]:\n"
            return processed_texts
        elif len(retrieved_facts) == 1:
            processed_texts = "Please acknowledge the updated information provided below and respond to the subsequent question.\n\n[Updated Information]:\n" \
            + retrieved_facts[0] + "\n\n[Question]:\n"
            return processed_texts
        else:
            return None

    def forward(
            self,
            input_ids,
            attention_mask,
            lora: bool,
            **kwargs
        ):
        if ((lora and not self.knowledge_base.is_empty()) or (not self.self_adaptive_post_alignment_inference)) and self.alignment:
            output = self.lora_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            with self.lora_model.disable_adapter():
                output = self.lora_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
        return output
        
    def generate(
            self,
            input_ids,
            attention_mask,
            max_new_tokens,
            lora: bool,
            **kwargs
        ):
        if ((lora and not self.knowledge_base.is_empty()) or (not self.self_adaptive_post_alignment_inference)) and self.alignment:
            output = self.lora_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                generation_config=GenerationConfig(
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                ),
                **kwargs
            )
        else:
            with self.lora_model.disable_adapter():
                output = self.lora_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    generation_config=GenerationConfig(
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    ),
                    **kwargs
                )
        return output
