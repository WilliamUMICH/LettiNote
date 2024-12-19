from sentence_transformers import SentenceTransformer
from sentence_transformers.util import *

from datasets import Dataset
import json, os, torch
import numpy as np
from tqdm import tqdm
from pprint import pprint

# ====
# Initial Flag Config
# ====
createEmbeddings = True
#NOTE: if set fales, will try to load data if already created.

# ========
# Create Datasets
# ========

# === original CN datasets ===
def myGen_original_cn():
    # data_path = '/Users/williamzheng/Documents/UmichFolder/2025 Fall Semester /Research 499/LettiNote/data/summaries/augmented_notes_small.jsonl'

    data_path = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/summaries/augmented_notes_small.jsonl'

    file = open(data_path)
    for r in file:
        yield json.loads(r)

original_cn_ds = Dataset.from_generator(myGen_original_cn)

# === medinote CN dataset ===
def myGen_medinote_cn():
    # data_path = '/Users/williamzheng/Documents/UmichFolder/2025 Fall Semester /Research 499/LettiNote/data/inference/mediNote-direct.jsonl'

    data_path = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/mediNote-direct.jsonl'

    file = open(data_path)
    for r in file:
        yield json.loads(r)

medinote_cn_ds = Dataset.from_generator(myGen_medinote_cn)

# print('Sample Original CN:\n', original_cn_ds[0])
# print('Sample Medinote CN:\n', medinote_cn_ds[0])
# print('Sample Original CN:\n', original_cn_ds[:3]['note'])



# save_path = '/Users/williamzheng/Documents/UmichFolder/2025 Fall Semester /Research 499/LettiNote/data/embeddings/Gold_CN'
# ds.save_to_disk(save_path)

# ========
# Create Embeddings
# ========

print(
    'Keys:',
    '\n   medinote keys: ',
    medinote_cn_ds.column_names,
    '\n   original keys: ',
    original_cn_ds.column_names
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
print('start encoding')

embeddingPath = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/embeddings/'

# === Orginal CN embedding ===
# TODO: create save and load dataset, check if created
original_cn_path = embeddingPath + 'original_cn/original_embed.pt'
if createEmbeddings:
    print('creating original embeddings')
    if os.path.exists(original_cn_path):
        os.remove(original_cn_path)

    original_cn_embedding = model.encode(
        original_cn_ds['note'], 
        show_progress_bar=True, 
        precision='int8',
        )
    
    torch.save(original_cn_embedding, original_cn_path)
else:
    try: 
        original_cn_embedding = torch.load(original_cn_path)
    except:
        print('Generate embeddings first!', original_cn_path)
    
# === Orginal MediNote embedding ===
# TODO: create save and load dataset, check if created
medinote_cn_path = embeddingPath + 'medinote_cn/medinote_embed.pt'

if createEmbeddings:
    print('creating medinote embeddings')
    if os.path.exists(medinote_cn_path):
        os.remove(medinote_cn_path)

    medinote_cn_embedding = model.encode(
        medinote_cn_ds['pred_direct'], #NOTE: MAKE SURE TO USE CORRECT KEY
        show_progress_bar=True, 
        precision='int8'
        )
    
    
    # print('Type: ', type(medinote_cn_embedding))
    # print(medinote_cn_embedding[:2])
    # print(medinote_cn_ds['note'][:2])

    torch.save(medinote_cn_embedding, medinote_cn_path)
else:
    try: 
        medinote_cn_embedding = torch.load(medinote_cn_path)
    except:
        print('Generate embeddings first!', medinote_cn_path)
        

# ========
# Semantic Search
# ========

#NOTE: need to convert back to correct datatype to work
medinote_cn_embedding = medinote_cn_embedding.astype('float32')
original_cn_embedding = original_cn_embedding.astype('float32')

results = semantic_search(
    medinote_cn_embedding[0],
    original_cn_embedding, 
    top_k=2 
)

pprint(results)
print('MEDINOTE QUERY')
# pprint(medinote_cn_ds[0]['pred_direct'])
print('FIRST BEST')
# pprint(original_cn_ds[results[0][0]['corpus_id']]['note'])
print('SECOND BEST')
# pprint(original_cn_ds[results[0][1]['corpus_id']]['note'])