import os
import os.path as path
import sys
import json
import torch
sys.path.append('..')
from easyeditor import (
    AlphaEditHyperParams,
    IKEHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    R_ROMEHyperParams,
    GraceHyperParams,
    WISEHyperParams,
    ERENHyperParams,
    KEDASHyperParams,
    LTEHyperParams,
    )
from easyeditor import CRAFTEditor, IKEEditorCRAFT, ERENEditorCRAFT, RECIPEEditorCRAFT, KEDASEditorCRAFT
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset, ZsreDataset
from knowedit_utils import process_knowedit, eval

import argparse
import numpy as np

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--train_data_path', default='../../data/ZsRE/zsre_mend_train_10000.json', type=str)
    parser.add_argument('--pre_file', default='./seq_pre.json', type=str)
    parser.add_argument('--teacher_forcing', required=True, type=int)
    parser.add_argument('--sequential_edit', required=True, type=int)
    args = parser.parse_args()

    if args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'EREN':
        editing_hparams = ERENHyperParams
    elif args.editing_method == 'LTE':
        editing_hparams = LTEHyperParams
    elif args.editing_method == 'KEDAS':
        editing_hparams = KEDASHyperParams
    else:
        raise NotImplementedError

    # datas = KnowEditDataset(args.data_dir,size=args.ds_size)
    # prompts, subjects, target_new, locality_inputs, portability_inputs = process_knowedit(args.datatype, datas)
    requests = json.load(open(args.data_dir,'r'))
    if args.ds_size is not None and args.ds_size > 0:
        requests = requests[:args.ds_size]
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    data_name = args.data_dir.split('/')[-1].split('.')[0]

    if args.ds_size is not None:
        # if args.editing_method in ['KEDAS', 'LTE', 'IKE', 'SERAC', 'MEND']:
        args.pre_file = f"./cache/{args.editing_method}_{data_name}_{hparams.model_name.split('/')[-1]}_pre_edit_{args.teacher_forcing}_{args.ds_size}.json"
        # else:
            # args.pre_file = f"./cache/{hparams.model_name.split('/')[-1]}_{args.datatype}_pre_edit_{args.ds_size}.json"
    else:
        # if args.editing_method in ['KEDAS', 'LTE', 'IKE', 'SERAC', 'MEND']:
        args.pre_file = f"./cache/{args.editing_method}_{data_name}_{hparams.model_name.split('/')[-1]}_pre_edit_{args.teacher_forcing}.json"
        # else:
            # args.pre_file = f"./cache/{hparams.model_name.split('/')[-1]}_{args.datatype}_pre_edit.json"
    print(args.pre_file)
    if args.pre_file is not None and os.path.exists(args.pre_file):
        pre_edit = json.load(open(args.pre_file,'r'))
        assert len(pre_edit) == len(requests)
    else:
        pre_edit = None

    if args.editing_method == 'IKE':
        train_ds = ZsreDataset(args.train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    elif args.editing_method == 'ICE':
        hparams.use_icl_examples = False
        train_ds = None
    else:
        train_ds = None

    vanilla_generation = (args.teacher_forcing == 0)
    keep_original_weight = (args.sequential_edit == 0)
    sequential_edit = (args.sequential_edit == 1)
    incremental_edit = (args.sequential_edit == 2)
    
    if args.editing_method == 'EREN':
        editor = ERENEditorCRAFT.from_hparams(hparams)
    elif args.editing_method == 'KEDAS' or args.editing_method == 'LTE':
        editor = KEDASEditorCRAFT.from_hparams(hparams)
    elif args.editing_method == 'IKE':
        editor = IKEEditorCRAFT.from_hparams(hparams)
    else:
        editor = CRAFTEditor.from_hparams(hparams)

    metrics, edited_model, _, edit_time, evaluation_time = editor.edit(
        requests=requests,
        train_ds=train_ds,
        keep_original_weight=keep_original_weight,
        pre_file=args.pre_file,
        pre_edit = pre_edit,
        test_generation=True,
        verbose=False,
        vanilla_generation=vanilla_generation,
        sequential_edit=sequential_edit,
        incremental_edit=incremental_edit,
    )

    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{data_name}_{hparams.model_name.split("/")[-1]}_results.json')
    json.dump(metrics, open(result_path, 'w'), indent=4)

    edit_hours = edit_time / 3600
    evaluation_hours = evaluation_time / 3600
    print(f"Edit time: {edit_hours:.3f} hours, Evaluation time: {evaluation_hours:.3f} hours")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")

    eval(result_path)
