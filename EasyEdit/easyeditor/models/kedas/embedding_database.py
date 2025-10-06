import torch
from sentence_transformers import SentenceTransformer

class MemoryElement:
    def __init__(self, id, type, key, value):
        self.id = id
        self.type = type
        self.key = key
        self.value = value

    def get_id(self):
        return self.id

    def get_type(self):
        return self.type

    def get_key(self):
        return self.key
    
    def get_value(self):
        return self.value


class EmbeddingDatabase:
    def __init__(self, hparams):
        self.device_name = f'cuda:{hparams.device}' if torch.cuda.is_available() and hparams.device >= 0 else 'cpu'
        self.sentence_model = SentenceTransformer(hparams.sentence_model_name)
        self.sentence_model.to(self.device_name)
        self.search_top_k = hparams.search_top_k
        self.dist_fn = hparams.dist_fn
        self.data = []
        self.keys = torch.tensor([])

    def clear(self):
        self.data = []
        self.keys = torch.tensor([])

    def get_data(self, index, type):
        for element in self.data:
            if element.get_id() == index and element.get_type() == type:
                return element.get_value()
        raise IndexError(f"Index {index} not found in the database.")
    
    def get_element(self, index, type):
        for element in self.data:
            if element.get_id() == index and element.get_type() == type:
                return element
        raise IndexError(f"Index {index} not found in the database.")
    
    def get_data_key(self, index, type):
        for element in self.data:
            if element.get_id() == index and element.get_type() == type:
                return element.get_key()
        raise IndexError(f"Index {index} not found in the database.")

    def add(self, id, type, value):
        if value == "" or not value:
            return
        
        key = self.sentence_model.encode(value, show_progress_bar=False, convert_to_tensor=True).detach().cpu()

        new_element = MemoryElement(id, type, key, value)
        self.data.append(new_element)
        self.keys = torch.cat((self.keys, key.unsqueeze(0)), dim=0)

    def search(self, query, k=None):
        query = self.sentence_model.encode(query, show_progress_bar=False, convert_to_tensor=True).detach().cpu()

        if len(self.data) == 0:
            return []

        nearest_elements = self._get_nearest(query, k=k if k is not None else self.search_top_k)
        
        results = []
        for nearest_element in nearest_elements:
            id = nearest_element["element"].get_id()
            type = nearest_element["element"].get_type()
            results.append((id, type))
        
        return results
    
    def to_string(self, type=None):
        if type is None:
            return [element.get_value() for element in self.data]
        else:
            return [(element.get_id(), element.get_value()) for element in self.data if element.get_type() == type]
    
    def get_nearest_qas(self, query, result_list, k=1):
        if len(result_list) == 0:
            return None
        
        query = self.sentence_model.encode(query, show_progress_bar=False, convert_to_tensor=True).detach().cpu()
        # keys = torch.stack([self.get_data_key(id, "request") for id in id_list])
        # keys = torch.stack([self.sentence_model.encode(result, show_progress_bar=False, convert_to_tensor=True).detach().cpu() for result in result_list])
        keys = torch.stack([self.get_data_key(id, type) for id, type in result_list])

        if self.dist_fn == "euc":
            distances = self._euclidean_distances_given_keys(query, keys)
        elif self.dist_fn == "arccos":
            distances = self._arccosine_distances_given_keys(query, keys)
        else:
            raise ValueError(f"Unknown distance function: {self.dist_fn}")
        
        # First get top 3*k elements by distance
        top_k = min(3 * k, len(result_list))
        top_distances, top_indices = torch.topk(distances, k=top_k, largest=False)
        
        # Create list of (distance, index, id) tuples for top elements
        elements_with_distances = []
        for i in range(top_k):
            idx = top_indices[i].item()
            elements_with_distances.append({
                "distance": top_distances[i].item(),
                "index": idx,
                "id": result_list[idx][0]
            })
        
        # Sort first by distance, then by ID in descending order (newer first)
        elements_with_distances.sort(key=lambda x: (x["distance"], -x["id"]))
        
        # Get unique IDs in order, keeping only the first occurrence of each ID
        top_ids = []
        seen_ids = set()
        for element in elements_with_distances:
            current_id = element["id"]
            if current_id not in seen_ids:
                top_ids.append(current_id)
                seen_ids.add(current_id)
            if len(top_ids) >= k:
                break

        return [self.get_data(id, "Q-A") for id in top_ids], top_ids
    
    def analyze(self, query, id):
        query = self.sentence_model.encode(query, show_progress_bar=False, convert_to_tensor=True).detach().cpu()
        if len(self.data) == 0:
            raise ValueError("No data in the database.")
        nearest_elements = self._get_nearest(query, k=len(self.data))
        retrieved_id = nearest_elements[0]["element"].get_id()
        retrieved_value = nearest_elements[0]["element"].get_value()
        return retrieved_id == id, retrieved_value

    def _euclidean_distances(self, query):
        return torch.cdist(query.unsqueeze(0), self.keys, p=2).squeeze(0)
    
    def _euclidean_distances_given_keys(self, query, keys):
        return torch.cdist(query.unsqueeze(0), keys, p=2).squeeze(0)
    
    def _arccosine_distances(self, query):
        query = query / torch.norm(query)
        keys = self.keys / torch.norm(self.keys, dim=0, keepdim=True)
        cosine_similarities = torch.mm(query.unsqueeze(0), keys.t()).squeeze(0)
        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
        return torch.acos(cosine_similarities)
    
    def _arccosine_distances_given_keys(self, query, keys):
        query = query / torch.norm(query)
        keys = keys / torch.norm(keys, dim=0, keepdim=True)
        cosine_similarities = torch.mm(query.unsqueeze(0), keys.t()).squeeze(0)
        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)
        return torch.acos(cosine_similarities)
    
    def _get_nearest(self, query, k=1):
        if len(self.data) == 0:
            return []

        if self.dist_fn == "euc":
            distances = self._euclidean_distances(query)
        elif self.dist_fn == "arccos":
            distances = self._arccosine_distances(query)
        else:
            raise ValueError(f"Unknown distance function: {self.dist_fn}")
        
        # First get top-3k elements by distance
        top_k = min(3 * k, len(self.data))
        top_distances, top_indices = torch.topk(distances, k=top_k, largest=False)
        
        # Create list of (distance, element_index, element) tuples only for top-k elements
        elements_with_distances = []
        for i in range(top_k):
            idx = top_indices[i].item()
            elements_with_distances.append({
                "distance": top_distances[i].item(),
                "element_index": idx,
                "element": self.data[idx]
            })
        
        # Sort first by distance, then by element ID in descending order (newer first)
        elements_with_distances.sort(key=lambda x: (x["distance"], -x["element"].get_id()))
        
        return elements_with_distances