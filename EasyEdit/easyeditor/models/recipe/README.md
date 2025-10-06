# RECIPE: Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning

This directory contains the implementation of RECIPE (Retrieval-Augmented Continuous Prompt Learning) adapted for the EasyEdit framework.

## Overview

RECIPE is a knowledge editing method that enables lifelong knowledge editing for Large Language Models (LLMs) through retrieval-augmented continuous prompt learning. The method was originally presented in the EMNLP 2024 paper:

> **Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning**
> 
> Authors: Qizhou Chen, Taolin Zhang, Xiaofeng He, Dongyang Li, Chengyu Wang, Longtao Huang, Hui Xue

## Key Features

- **Retrieval-Augmented Learning**: Uses a knowledge representation model to encode and retrieve relevant knowledge
- **Continuous Prompt Learning**: Generates dynamic prompts based on retrieved knowledge representations
- **Lifelong Editing**: Supports sequential editing without catastrophic forgetting
- **Batch Editing**: Can handle multiple edits simultaneously

## Architecture Components

### 1. Knowledge Representation Model (KRM)
- Based on RoBERTa for encoding knowledge and queries
- Produces dense representations for both knowledge and queries
- Includes separate transformation paths for knowledge vs. query encoding

### 2. Prompt Transformer
- Converts knowledge representations into editing prompts
- Generates continuous prompts that are injected into the model's forward pass
- Uses multi-layer perceptrons with residual connections

### 3. Retrieval System
- Computes cosine similarity between query and knowledge representations
- Retrieves top-k most relevant knowledge pieces for each query
- Supports configurable similarity thresholds

## Files

- `recipe_main.py`: Main interface functions (`apply_recipe_to_model` and `execute_recipe`)
- `recipe.py`: Core RECIPE class implementation
- `recipe_hparams.py`: Hyperparameter configuration class following EasyEdit conventions
- `recipe_models.py`: Core model components (KnowledgeRepModel and PromptTransformer)
- `__init__.py`: Module exports

## Usage

### Basic Usage

```python
from easyeditor import BaseEditor, RECIPEHyperParams

# Load hyperparameters
hparams = RECIPEHyperParams.from_hparams('./hparams/RECIPE/llama-7b.yaml')

# Initialize editor
editor = BaseEditor.from_hparams(hparams)

# Perform editing
prompts = ["The capital of France is"]
target_new = ["Lyon"]
ground_truth = ["Paris"]

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    target_new=target_new,
    ground_truth=ground_truth
)
```

### Configuration

The method supports the following key hyperparameters:

- `prompt_token_n`: Number of prompt tokens to generate
- `knowledge_rep_dim`: Dimension of knowledge representations
- `knowl_rep_prot_token_n`: Number of prototype tokens
- `model_hidden_size`: Hidden size of the target model
- `begin_layer_path`: Path to the layer where prompts are injected
- `lm_head_path`: Path to the language model head
- Various training hyperparameters (learning rates, loss weights, etc.)

### Supported Models

RECIPE has been tested with:
- Llama-2-7B
- GPT-J-6B  
- GPT2-XL

Configuration files are provided in `hparams/RECIPE/` for each supported model.

## Training

Unlike methods that modify model parameters directly, RECIPE requires training the knowledge representation model and prompt transformer components. The training process involves:

1. **Reliability Loss**: Ensures the model produces correct outputs for edited facts
2. **Generalization Loss**: Maintains performance on paraphrases and related queries
3. **Locality Loss**: Preserves performance on unrelated knowledge
4. **Contrastive Loss**: Improves knowledge representation quality

## Dependencies

- PyTorch
- Transformers (HuggingFace)
- RoBERTa model for knowledge representation

## Citation

If you use this implementation, please cite the original RECIPE paper:

```bibtex
@inproceedings{chen2024recipe,
    title = {Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning},
    author={Qizhou Chen and Taolin Zhang and Xiaofeng He and Dongyang Li and Chengyu Wang and Longtao Huang and Hui Xue},
    year = 2024,
    booktitle = {EMNLP},
    url = {https://2024.emnlp.org/program/accepted_main_conference/}
}
``` 