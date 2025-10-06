import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

class RelevanceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        request = item['request'][:200]
        knowledge = item['knowledge'][:200]
        relevance = item['relevance']

        encoding = self.tokenizer.encode_plus(
            knowledge,
            request,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(relevance, dtype=torch.long)
        }

class RelevanceFilter:
    def __init__(self, model_path='./hugging_cache/bert-base-uncased', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'roberta' in model_path.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        # else:
        #     raise ValueError(f"Unsupported model: {model_path}")
        
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, hparams):
        model_path = hparams.filter_model_name
        device_name = f'cuda:{hparams.device}' if torch.cuda.is_available() and hparams.device >= 0 else 'cpu'
        return cls(model_path, device_name)

    def save_model(self, path):
        """Save the model and tokenizer to the specified path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        """Load the model and tokenizer from the specified path."""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)

    def predict(self, request, knowledge):
        """Predict the relevance between a request and knowledge."""
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            knowledge[:200],
            request[:200],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            
        return predictions.item()
    
    def predict_with_confidence(self, request, knowledge):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            knowledge[:200],
            request[:200],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
        
        prediction_confidence = torch.softmax(logits, dim=1).tolist()[0]
        prediction_confidence = prediction_confidence[predictions.item()]
        return predictions.item(), prediction_confidence