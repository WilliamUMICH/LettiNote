import json, os, pandas as pd, pprint
from datasets import Dataset

def myGen_original_cn(data_path):
    file = open(data_path)
    for r in file:
        yield json.loads(r)

if False:
    file = open('/home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/gpt3-mysum-CN.jsonl')
    for i in file:
        pprint.pprint(json.loads(i))
        break

if False:
    # ====
    # prediction
    # ====
    data_path = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/my_mediNote-PS.jsonl'
    pred_ps_ds = Dataset.from_generator(
        myGen_original_cn, 
        gen_kwargs={'data_path': data_path}
    )
    pprint.pprint(pred_ps_ds['pred_summary'][0])
    print('\n\n')
    pprint.pprint(pred_ps_ds['pred_summary'][1])
    print('\n\n')
    json_pred_summary = pred_ps_ds['pred_summary'][0]
    json_pred_summary = json.loads(json_pred_summary)

    print(json_pred_summary.keys())

    json_pred_summary = pred_ps_ds['pred_summary'][1]
    json_pred_summary = json.loads(json_pred_summary)

    print(json_pred_summary.keys())

    # ====
    # augmented
    # ====

    data_path = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/summaries/augmented_notes_small.jsonl'
    ps_ds = Dataset.from_generator(
        myGen_original_cn, 
        gen_kwargs={'data_path': data_path}
    )
    pprint.pprint(ps_ds['summary'][0])
    print('\n\n')
    pprint.pprint(ps_ds['summary'][1])
    print('\n\n')
    json_summary = ps_ds['summary'][0]
    json_summary = json.loads(json_summary)

    print(json_summary)


if False:
    # ====
    # augmented
    # ====

    # LettiNote/MediNote/data/summaries/augmented_notes_30K.jsonl
    

    original_cn_ds = Dataset.from_generator(myGen_original_cn)

    # print('FIRST: ', original_cn_ds['summary'][0])
    initial_json_summary = original_cn_ds['summary'][0]
    initial_json_summary = str.replace(initial_json_summary, '""""', '"')
    # print('FIRST: ', initial_json_summary)

    initial_json_summary = json.loads(initial_json_summary)

    for i in range(1, 10):
        json_summary = original_cn_ds['summary'][i]
        print('FIRST:\n', json_summary)
        json_summary = str.replace(json_summary, '""""None""""', '')
        print('FIRST:\n', json_summary)
        json_summary = json.loads(json_summary)
        

        if initial_json_summary.keys() == json_summary.keys():
            print('CHECKING')
            print('FIRST: ', 
                  json_summary.keys(), '\n')
            print(True)
        else:
            print(False)


# LettiNote/MediNote/data/summaries/augmented_notes_30K.jsonl

def myGen_original_cn():
    data_path = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/summaries/augmented_notes_30K.jsonl'
    file = open(data_path)
    for r in file:
        yield json.loads(r)

original_cn_ds = Dataset.from_generator(myGen_original_cn)

dataset_no_summary = original_cn_ds.select_columns(['note', 'conversation', 'idx', 'full_note'])
path = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/misc/full_no_PS.jsonl'
print(dataset_no_summary.to_json(path))
print(dataset_no_summary.column_names)


