#!/bin/bash
set -euxo pipefail
export CUDA_VISIBLE_DEVICES=2

BASELINES=("LTE")
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

BASELINES=("KEDAS")
MODEL="llama3-8b"
TEACHER_FORCING=1
SEQUENTIAL_EDIT=1

for BASELINE in ${BASELINES[@]}; do
    SUFFIX=${TEACHER_FORCING}_${SEQUENTIAL_EDIT}
    RESULT_DIR=output/${MODEL}/${BASELINE}_${SUFFIX}
    mkdir -p ${RESULT_DIR}

    HPARAMS_DIR=../hparams/${BASELINE}/${MODEL}-S.yaml
    nohup setsid python run_CRAFT.py \
        --editing_method ${BASELINE} \
        --hparams_dir ${HPARAMS_DIR} \
        --data_dir ../../data/CRAFT/CRAFT-Statistical-test500.json \
        --metrics_save_dir ${RESULT_DIR} \
        --teacher_forcing $TEACHER_FORCING \
        --sequential_edit $SEQUENTIAL_EDIT \
        > ${RESULT_DIR}/CRAFT-S_${BASELINE}_${SUFFIX}.txt

    HPARAMS_DIR=../hparams/${BASELINE}/${MODEL}-F.yaml
    nohup setsid python run_CRAFT.py \
        --editing_method ${BASELINE} \
        --hparams_dir ${HPARAMS_DIR} \
        --data_dir ../../data/CRAFT/CRAFT-Financial-test500.json \
        --metrics_save_dir ${RESULT_DIR} \
        --teacher_forcing $TEACHER_FORCING \
        --sequential_edit $SEQUENTIAL_EDIT \
        > ${RESULT_DIR}/CRAFT-F_${BASELINE}_${SUFFIX}.txt
done
