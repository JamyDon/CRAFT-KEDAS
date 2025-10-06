import json
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add the EasyEdit directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from EasyEdit.easyeditor.models.kedas.filter import RelevanceFilter, RelevanceDataset

def train_model(mode, model_name, model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    """Train the model with the given data loaders."""
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Create checkpoint directory
    checkpoint_dir = Path(f"../../checkpoints/filter/{mode}_{model_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training
        model.model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            model.model.zero_grad()
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Average training loss: {avg_train_loss:.4f}')

        # Validation
        model.model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)

                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_model(checkpoint_dir / 'best_model')
            print(f'Saved best model to {checkpoint_dir / "best_model"}')

def main():
    # Load data
    mode = 'CRAFT'
    model_name = 'bert'
    data_path = Path(f"../../data/filter/{mode}.json")
    model_path = "./hugging_cache/bert-base-chinese"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    # Initialize model and create datasets
    filter_model = RelevanceFilter(model_path=model_path)
    train_dataset = RelevanceDataset(train_data, filter_model.tokenizer)
    val_dataset = RelevanceDataset(val_data, filter_model.tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train the model
    train_model(mode, model_name, filter_model, train_loader, val_loader)

    # Save the final model
    checkpoint_dir = Path(f"../../checkpoints/filter/{mode}_{model_name}")
    filter_model.save_model(checkpoint_dir / 'final_model')
    print(f'Saved final model to {checkpoint_dir / "final_model"}')

if __name__ == "__main__":
    main()
