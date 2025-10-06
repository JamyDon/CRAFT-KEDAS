import json
import torch
import numpy as np
import random
from tqdm import tqdm

from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationConfig
from sentence_transformers import SentenceTransformer, util
from typing import Optional

class IKEModel(PreTrainedModel):
    def __init__(
            self,
            model,
            tokenizer,
            sentence_model,
            hparams
        ):
        super().__init__(model.config)
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.model = model
        self.sentence_model = sentence_model
        self.device_name = f'cuda:{hparams.device}' if torch.cuda.is_available() and hparams.device >= 0 else 'cpu'

        self.training_data = self.load_training_data(hparams.training_data_path)
        
        self.k = hparams.k
        self.embeddings = np.array([])
        self.facts = []

        self.model.to(self.device_name)
        if self.sentence_model:
            self.sentence_model.to(self.device_name)

    @classmethod
    def from_pretrained(cls, 
                      base_model_path: str, 
                      sentence_model_path: Optional[str] = None,
                      **kwargs):
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)

        sentence_model = None
        if sentence_model_path:
            sentence_model = SentenceTransformer(sentence_model_path)
        
        return cls(model, sentence_model, kwargs)
    
    def load_training_data(self, training_data_path):
        with open(training_data_path, 'r') as f:
            data = json.load(f)
        training_data = {'sentence': [], 'embedding': []}
        # training_data = {'copy': {'sentence': [], 'embedding': []}, 'update': {'sentence': [], 'embedding': []}, 'retain': {'sentence': [], 'embedding': []}}
        for d in tqdm(data, desc="Loading training data"):
            src = d['src']
            rephrase = d['rephrase']
            alt = d['alt']
            loc = d['loc'].split('nq question:')[1].strip()
            loc_ans = d['loc_ans']
            new_fact = f"{src} {alt}"

            copy_sample = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
            copy_sample_vector = self.sentence_model.encode(copy_sample, show_progress_bar=False)
            update_sample = f"New Fact: {new_fact}\nPrompt: {rephrase} {alt}\n\n"
            update_sample_vector = self.sentence_model.encode(update_sample, show_progress_bar=False)
            retain_sample = f"New Fact: {new_fact}\nPrompt: {loc}?` {loc_ans}\n\n"
            retain_sample_vector = self.sentence_model.encode(retain_sample, show_progress_bar=False)

            # training_data['copy']['sentence'].append(copy_sample)
            # training_data['update']['sentence'].append(update_sample)
            # training_data['retain']['sentence'].append(retain_sample)
            # training_data['copy']['embedding'].append(copy_sample_vector)
            # training_data['update']['embedding'].append(update_sample_vector)
            # training_data['retain']['embedding'].append(retain_sample_vector)

            training_data['sentence'].append(copy_sample)
            training_data['sentence'].append(update_sample)
            training_data['sentence'].append(retain_sample)
            training_data['embedding'].append(copy_sample_vector)
            training_data['embedding'].append(update_sample_vector)
            training_data['embedding'].append(retain_sample_vector)

        return training_data
    
    def edit(self, query, new_fact, keep_original_weight=False):
        query_vector = self.sentence_model.encode(query, show_progress_bar=False)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        self.embeddings = np.concatenate((self.embeddings, query_vector), axis=0) if self.embeddings.size else query_vector
        self.facts.append(new_fact)

        if keep_original_weight:
            # keep the last edit only
            self.embeddings = self.embeddings[-1:]
            self.facts = self.facts[-1:]
        
    def sentence_retrieval(
        self,
        query_sentence: str,
    ):
        stored_embeddings = torch.tensor(self.embeddings)
        stored_embeddings = util.normalize_embeddings(stored_embeddings)

        query_embedding = util.normalize_embeddings(torch.tensor(self.sentence_model.encode(
            query_sentence, show_progress_bar=False)).unsqueeze(0))

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=1)
        assert len(hits) == 1
        hit = hits[0]
        retrieved_sentences = [self.facts[hit[k]["corpus_id"]] for k in range(len(hit))]
        
        return retrieved_sentences[0]

    def training_data_retrieval(
        self,
        sentence: str,
    ):
        stored_embeddings = np.array(self.training_data['embedding'])
        # stored_embeddings = torch.tensor(self.training_data['embedding'])
        stored_embeddings = torch.from_numpy(stored_embeddings)
        stored_embeddings = util.normalize_embeddings(stored_embeddings)

        query_embedding = util.normalize_embeddings(torch.tensor(self.sentence_model.encode(
            sentence, show_progress_bar=False)).unsqueeze(0))

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=self.k)
        assert len(hits) == 1
        hit = hits[0]
        retrieved_sentences = [self.training_data['sentence'][hit[k]["corpus_id"]] for k in range(len(hit))]

        return retrieved_sentences
        
    def training_data_retrieval_one(
        self,
        sentence: str,
        type: str,
        topk: int,
    ):
        stored_embeddings = torch.tensor(self.training_data[type]['embedding'])
        stored_embeddings = util.normalize_embeddings(stored_embeddings)

        query_embedding = util.normalize_embeddings(torch.tensor(self.sentence_model.encode(
            sentence, show_progress_bar=False)).unsqueeze(0))

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=topk)
        assert len(hits) == 1
        hit = hits[0]
        retrieved_sentences = [self.training_data[type]['sentence'][hit[k]["corpus_id"]] for k in range(len(hit))]

        return retrieved_sentences
    
    def training_data_retrieval_all(
        self,
        sentence: str,
    ):
        copy_sentences = self.training_data_retrieval_one(sentence, 'copy', self.k // 8)
        update_sentences = self.training_data_retrieval_one(sentence, 'update', 3 * self.k // 8)
        retain_sentences = self.training_data_retrieval_one(sentence, 'retain', self.k // 2)
        sentences = copy_sentences + update_sentences + retain_sentences
        
        np.random.shuffle(sentences)

        return sentences
    
    def retrieve_prefix(
        self,
        input_text: str,
    ):
        retrieved_fact = self.sentence_retrieval(input_text)
        input_text = f"New Fact: {retrieved_fact}\nPrompt: {input_text}\n\n"
        # retrieved_training_data = self.training_data_retrieval_all(input_text)
        retrieved_training_data = self.training_data_retrieval(input_text)
        retrieved_training_data.append(f"New Fact: {retrieved_fact}\nPrompt:")
        processed_text = "".join(retrieved_training_data)

        # print(f"Input text: {input_text}")
        # print(f"Retrieved fact: {retrieved_fact}")
        # print(f"Retrieved training data: {retrieved_training_data}")
        # print(f"Processed text: {processed_text}")
        # print('-' * 100)
                
        return processed_text

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
