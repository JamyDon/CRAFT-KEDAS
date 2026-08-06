# CRAFT & KEDAS

[![arXiv](https://img.shields.io/badge/arXiv-2508.01302-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2508.01302)
[![Hugging Face Collection](https://img.shields.io/badge/CRAFT_Dataset-3B4252?style=flat&logo=huggingface)](https://huggingface.co/datasets/JamyDohrn/CRAFT)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache)](LICENSE)

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
The CRAFT dataset used for experiments in our paper is located in `data/CRAFT`, which is constructed during 25Q1.

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

## Citation
If you find our work useful, feel free to cite our paper:
```bib
@inproceedings{tang-etal-2026-aligning,
    title = "Aligning Language Models with Real-time Knowledge Editing",
    author = "Tang, Chenming  and
      Yang, Yutong  and
      Wang, Kexue  and
      Wu, Yunfang",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.14/",
    doi = "10.18653/v1/2026.acl-long.14",
    pages = "363--378",
    ISBN = "979-8-89176-390-6"
}
```
