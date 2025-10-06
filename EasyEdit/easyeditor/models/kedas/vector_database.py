import torch


class MemoryElement:
    def __init__(self, id, key, value, radius, hidden=False, hidden_by=None):
        self.id = id
        self.key = key
        self.value = value
        self.radius = radius
        self.hidden = hidden
        self.hidden_by = hidden_by

    def get_id(self):
        return self.id

    def get_key(self):
        return self.key
    
    def get_value(self):
        return self.value
    
    def get_radius(self):
        return self.radius
    
    def get_hidden(self):
        return self.hidden
    
    def get_hidden_by(self):
        return self.hidden_by

    def set_radius(self, radius):
        self.radius = radius

    def set_hidden(self, hidden):
        self.hidden = hidden

    def set_hidden_by(self, hidden_by):
        self.hidden_by = hidden_by


class VectorDatabase:
    def __init__(self, hparams):
        self.data = []
        self.keys = torch.tensor([])
        self.hidden_data = []
        self.initial_radius = hparams.initial_radius
        self.radius_constant = hparams.radius_constant
        self.adopt_radius = hparams.adopt_radius
        self.dist_fn = hparams.dist_fn
        self.search_top_k = hparams.search_top_k
        self.idx_counter = 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self.data):
            raise IndexError("Index out of range.")
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def add(self, key, value, return_id=False):
        key = torch.tensor(key)

        nearest_elements = self._get_nearest(key, k=1)
        nearest_element = nearest_elements[0] if len(nearest_elements) > 0 else None
        nearest_index = nearest_element["element_index"] if nearest_element else None
        nearest_distance = nearest_element["distance"] if nearest_element else None
        nearest_id = nearest_element["element"].get_id() if nearest_element else None
        nearest_value = nearest_element["element"].get_value() if nearest_element else None
        nearest_radius = nearest_element["element"].get_radius() if nearest_element else None

        if nearest_distance is None or nearest_distance > nearest_radius + self.initial_radius:
            new_element = MemoryElement(
                id=self.idx_counter,
                key=key,
                value=value,
                radius=self.initial_radius
            )
            self.idx_counter += 1
            self.data.append(new_element)
            self.keys = torch.cat((self.keys, key.unsqueeze(0)), dim=0)
        else:
            if nearest_value == value:
                self.data[nearest_index].set_radius(nearest_radius + self.initial_radius)
                new_element = MemoryElement(
                    id=self.idx_counter,
                    key=key,
                    value=value,
                    radius=0,
                    hidden=True,
                    hidden_by=nearest_id
                )
                self.idx_counter += 1
                self.hidden_data.append(new_element)
            else:
                self.data[nearest_index].set_radius(nearest_distance * (0.5 - self.radius_constant))
                new_element = MemoryElement(
                    id=self.idx_counter,
                    key=key,
                    value=value,
                    radius=nearest_distance * (0.5 - self.radius_constant),
                )
                self.idx_counter += 1
                self.data.append(new_element)
                self.keys = torch.cat((self.keys, key.unsqueeze(0)), dim=0)
        
        if return_id:
            return new_element.get_id()
        return None

    def search(self, query):
        query = torch.tensor(query)

        if len(self.data) == 0:
            return None

        nearest_elements = self._get_nearest(query, k=self.search_top_k)
        
        for nearest_element in nearest_elements:
            # return nearest_element["element"].get_value()
            radius = nearest_element["element"].get_radius()
            distance = nearest_element["distance"]

            if not self.adopt_radius:
                return nearest_element["element"].get_value()

            if radius >= distance:
                return nearest_element["element"].get_value()
            
        return None
    
    def analyze(self, query, id, value):
        # for retrieval evaluation
        # return whether the query is within the radius of the element at id
        # return the actually retrieved element
        query = torch.tensor(query)
        if len(self.data) == 0:
            raise ValueError("No data in the database.")
        nearest_elements = self._get_nearest(query, k=len(self.data) if self.adopt_radius else 1)
        retrieved = False
        retrieved_element = nearest_elements[0]["element"].get_value()
        for i, nearest_element in enumerate(nearest_elements):
            radius = nearest_element["element"].get_radius()
            distance = nearest_element["distance"]
            if nearest_element["element"].get_id() == id or value == nearest_element["element"].get_value():
                retrieved = True
                if not self.adopt_radius or radius >= distance:
                    return True, retrieved_element
        # if not retrieved:
        #     raise ValueError(f"Element with id {id} not found in the database.")
        return False, retrieved_element

    def _euclidean_distances(self, query):
        return torch.cdist(query.unsqueeze(0), self.keys, p=2).squeeze(0)
    
    def _arccosine_distances(self, query):
        query = query / torch.norm(query)
        keys = self.keys / torch.norm(self.keys, dim=1, keepdim=True)
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
        
        k = min(k, len(self.data))
        nearest_distances, indices = torch.topk(distances, k, largest=False)
        # import random
        # indices = random.sample(range(len(self.data)), k)
        # nearest_distances = distances[indices]
        nearest_elements = [{
            "element_index": j,
            "element": self.data[j],
            "distance": nearest_distances[i]
        } for i, j in enumerate(indices)]
        
        return nearest_elements