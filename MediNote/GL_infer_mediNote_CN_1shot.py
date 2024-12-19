from utils.infer import *
# from utils.data import *
# from utils.chat import *
# from utils.generate import *

# current file /home/willizhe/wz_stuff/LettiNote/MediNote/GL_infer_mediNote_CN_1shot.py

modelName = "llama"
# modelPath = '/scratch/jjcorso_root/jjcorso98/willizhe/model/medinote-7b'
# modelPath = '/home/willizhe/.cache/huggingface/hub/models--AGBonnet--medinote-7b'
modelPath = 'AGBonnet/medinote-7b'

inputPath = '/home/willizhe/wz_stuff/LettiNote/MediNote/data/misc/full_similarNote.jsonl'
outputPath = 'LettiNote/MediNote/data/inference/mediNote_200_1shot.jsonl'

numSamples = 200
# This parameter controls how notes to generate

# mode = "generator"
mode = "direct"

# line 511 in infer.py     generator -> generate notes from summarizer's summaries

infer(
    model_name=modelName,
    model_path=modelPath,
    input_path=inputPath,
    output_path=outputPath,
    num_samples=numSamples,
    mode=mode,
    template_path='LettiNote/MediNote/generation/templates/template.json',
    oneShot=True
)
