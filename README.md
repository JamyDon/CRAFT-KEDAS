# CRAFT & KEDAS

These are the data and source code for our ACL 2026 (main conference) paper: **Aligning Language Models with Real-time Knowledge Editing**.

The code is primarily based on the EasyEdit framework with some modifications to be compatible for some methods and settings.

## Prerequisites

### Environment
We primarily follow the environment of EasyEdit. Python 3.9.7 is recommended.
```bash
cd EasyEdit
pip install -r requirements.txt
```
### Data
We directly provide most of the data in `data`.
### Models
Please follow `EasyEdit/examples/hugging_cache/README.md` to download the models required.
### Others
An OpenAI API key is required if you want to reproduce the diverse edit augmentation. Note that we have provided an augmented version in `data/preprocessed_CRAFT`.
```bash
export OPENAI_API_KEY=<your-api-key>
```

## The CRAFT Dataset
The CRAFT dataset used for experiments in our paper is located in `data/CRAFT`.

For detailed data construction, please refer to `CRAFT/README.md`.

## Reproduce KEDAS

### Training Data Preparation
```bash
cd train/alignment
python create_train_data.py

cd ../filter
python prepare_data.py

cd ../../data/LTE
python convert.py
```

### Alignment
We use LLaMA-Factory. We recommend to employ a separate environment:
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
Then, run the alignment:
```bash
llamafactory-cli train ../train/alignment/llama3_lora.yaml
```

### Filter Training
```bash
python trian/filter/train.py
```

### Preprocessing of Diverse Edit Augmentation
```bash
cd EasyEdit/examples
python preprocess_knowledge.py
```

### Running KEDAS
```bash
cd EasyEdit/examples

BASELINES=("KEDAS")
MODEL="llama3-8b"
TEACHER_FORCING=1
SEQUENTIAL_EDIT=1

for BASELINE in ${BASELINES[@]}; do
    SUFFIX=${TEACHER_FORCING}_${SEQUENTIAL_EDIT}
    HPARAMS_DIR=../hparams/${BASELINE}/${MODEL}_CRAFT.yaml
    RESULT_DIR=output/${MODEL}/${BASELINE}_${SUFFIX}
    mkdir -p ${RESULT_DIR}

    nohup setsid python run_CRAFT.py \
        --editing_method ${BASELINE} \
        --hparams_dir ${HPARAMS_DIR} \
        --data_dir ../../data/CRAFT/CRAFT-Statistical-test500.json \
        --metrics_save_dir ${RESULT_DIR} \
        --teacher_forcing $TEACHER_FORCING \
        --sequential_edit $SEQUENTIAL_EDIT \
        > ${RESULT_DIR}/CRAFT-S_${BASELINE}_${SUFFIX}.txt

    nohup setsid python run_CRAFT.py \
        --editing_method ${BASELINE} \
        --hparams_dir ${HPARAMS_DIR} \
        --data_dir ../../data/CRAFT/CRAFT-Financial-test500.json \
        --metrics_save_dir ${RESULT_DIR} \
        --teacher_forcing $TEACHER_FORCING \
        --sequential_edit $SEQUENTIAL_EDIT \
        > ${RESULT_DIR}/CRAFT-F_${BASELINE}_${SUFFIX}.txt
done
```
