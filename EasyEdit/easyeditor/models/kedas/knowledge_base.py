import json

from .filter import RelevanceFilter

from .embedding_database import EmbeddingDatabase

class KnowledgeBase:
    def __init__(self, hparams):
        self.embedding_db = EmbeddingDatabase(hparams)
        self.filter_model = RelevanceFilter.from_pretrained(hparams)
        self.data_cache = self._load_data_cache(hparams)    # for experiment
        self.idx_counter = 0
        self.ike_top_k = hparams.ike_top_k
        self.editing_time_augmentation = hparams.editing_time_augmentation
        self.with_filter = hparams.with_filter

    def is_empty(self):
        return self.idx_counter == 0
    
    def clear(self):
        self.embedding_db.clear()
        self.idx_counter = 0

    def add(self, request, target_new):
        idx = self.idx_counter
        self.idx_counter += 1

        knowledge_item = self._search_in_cache(request)

        if not knowledge_item:
            self.embedding_db.add(idx, "request", request)
            self.embedding_db.add(idx, "Q-A", request + " " + target_new)
            return idx

        if self.editing_time_augmentation:
            self.embedding_db.add(idx, "request", knowledge_item["request"])
            self.embedding_db.add(idx, "Q-A", knowledge_item["Q-A"])
            self.embedding_db.add(idx, "Declaration", knowledge_item["Declaration"])
            self.embedding_db.add(idx, "Paraphrased", knowledge_item["Paraphrased"])
        else:
            self.embedding_db.add(idx, "Q-A", knowledge_item["Q-A"])

        return idx

    def search(self, query):
        embedding_results = self.embedding_db.search(query)
        results = embedding_results

        if len(results) == 0:
            return []
        
        final_ids, rejected_ids = [], []
        final_results = []
        for rid, typee in results:
            if rid in final_ids or rid in rejected_ids:
                continue
            qa = self.embedding_db.get_data(rid, "Q-A")

            if not self.with_filter:
                final_ids.append(rid)
                final_results.append((rid, typee))
                continue

            prediction, confidence = self.filter_model.predict_with_confidence(query, qa)
            if not prediction:
                rejected_ids.append(rid)
                continue
            final_ids.append(rid)
            final_results.append((rid, typee))

        if len(final_results) == 0:
            return []

        final_qa, final_ids = self.embedding_db.get_nearest_qas(query, final_results, k=self.ike_top_k)

        return final_qa

    def analyze(self, query, id, type=None):
        embedding_results = self.embedding_db.search(query)
        results = embedding_results

        final_ids, rejected_ids = [], []
        final_results = []
        for rid, typee in results:
            if rid in final_ids or rid in rejected_ids:
                continue
            qa = self.embedding_db.get_data(rid, "Q-A")
            prediction, confidence = self.filter_model.predict_with_confidence(query, qa)
            if not prediction:
                rejected_ids.append(rid)
                continue    
            final_ids.append(rid)
            final_results.append((rid, typee))

        if len(final_results) == 0:
            if type == 'prompt':
                print(f"query: {query}")
                print(f"id: {id}")
                print(f"final_results: {final_results}")
                print(f"final_ids: {final_ids}")
                # print(f"final_qa: {final_qa}")
                print(f"self.embedding_db.to_string('request'): {self.embedding_db.to_string('request')}")
                print('-'*20)
            return False, None

        final_qa, final_ids = self.embedding_db.get_nearest_qas(query, final_results, k=self.ike_top_k)

        if type == 'locality':
            return True, "True"

        if id in final_ids:
            return True, "True"

        return False, None

    def _load_data_cache(self, hparams):
        if hparams.data_cache_path is None:
            return []
        with open(hparams.data_cache_path, "r") as f:
            data_cache = json.load(f)
        return data_cache
    
    def _search_in_cache(self, query):
        for item in self.data_cache:
            if item["request"] == query:
                return item
        return None