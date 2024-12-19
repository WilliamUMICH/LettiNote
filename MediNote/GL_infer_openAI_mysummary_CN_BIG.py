from utils.infer import *
from utils.data import *
from utils.chat import *
from utils.generate import *

modelName = "gpt3"
mode = "mysum-CN" # :param mode: str, mode of inference (direct-gpt, generator-gpt)
max_tokens = 20000
num_samples = 30000

# inputPath = 'LettiNote/MediNote/data/misc/no_PS.jsonl'
inputPath = 'LettiNote/MediNote/data/misc/full_no_PS.jsonl'
outputPath = 'LettiNote/MediNote/data/inference/full_gpt3_mysum_CN.jsonl' 

# Generate clinical notes from conversations using an OpenAI model.
infer_openai(
    input_path=inputPath,
    output_path=outputPath,
    model_name=modelName,
    num_samples=num_samples,
    mode=mode,
    max_tokens=max_tokens
)

#NOTE: outcome is that clinical notes generated from conversation, i think information is lost