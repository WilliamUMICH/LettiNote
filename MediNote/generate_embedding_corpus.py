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


# [print(index) for index, i in enumerate(open('/home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/full_gpt3_mysum_CN.jsonl'))]

def load_dataset(path):

    file = open(path)
    for index, r in enumerate(file):
        yield json.loads(r)

# === original CN datasets ===
gen_arg = {
    'path' : '/home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/full_gpt3_mysum_CN.jsonl'
}
ds_mysum_CN = Dataset.from_generator(load_dataset, gen_kwargs=gen_arg)

print(ds_mysum_CN)
# ========
# Create Embeddings
# ========

print(gen_arg['path'])
print(
    'Keys:',
    '\n   ds_mysum_CN keys: ',
    ds_mysum_CN.column_names,
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
print('start encoding')

embeddingPath = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/embeddings/'

# === Orginal CN embedding ===
# TODO: create save and load dataset, check if created
ds_mysum_CN_path = embeddingPath + 'mysum/full_embed_mysum_CN.pt'
if createEmbeddings:
    print('creating original embeddings')
    if os.path.exists(ds_mysum_CN_path):
        os.remove(ds_mysum_CN_path)

    mysum_CN_embedding = model.encode(
        ds_mysum_CN['mysum-CN'], 
        show_progress_bar=True, 
        precision='int8',
        )
    
    torch.save(mysum_CN_embedding, ds_mysum_CN_path)
else:
    try: 
        mysum_CN_embedding = torch.load(ds_mysum_CN_path)
    except:
        print('Generate embeddings first!', ds_mysum_CN_path)
    


# ========
# Semantic Search
# ========

# NOTE: need to convert back to correct datatype to work
mysum_CN_embedding = mysum_CN_embedding.astype('float32')

results = semantic_search(
    mysum_CN_embedding,
    mysum_CN_embedding, 
    top_k=2 
)

# pprint(results)

print('\n\n\n\n')

print('CONTENT')

newColumnData = []
for index, top2 in tqdm(enumerate(results), desc='finding most similar'):
    most_similar_index = top2[1]['corpus_id']
    # print(most_similar_index)
    newColumnData.append(ds_mysum_CN['note'][most_similar_index])


ds_mysum_CN = ds_mysum_CN.add_column('most_similar_CN', newColumnData)
print(ds_mysum_CN)
ds_mysum_CN.to_json('/home/willizhe/wz_stuff/LettiNote/MediNote/data/misc/full_similarNote.jsonl',)
    