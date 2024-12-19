from utils.infer import *
from utils.data import *
from utils.chat import *
from utils.generate import *

modelName = "gpt3"
mode = "mysum-direct" # :param mode: str, mode of inference (direct-gpt, generator-gpt)
max_tokens = 20000
num_samples = 3750
inputPath = 'LettiNote/MediNote/data/misc/no_PS.jsonl'
outputPath = 'LettiNote/MediNote/data/inference/gpt3-mysum-direct.jsonl' 

# Generate clinical notes from conversations using an OpenAI model.
infer_openai(
    input_path=inputPath,
    output_path=outputPath,
    model_name=modelName,
    mode=mode,
    max_tokens=max_tokens,
    num_samples=num_samples
)

#NOTE: outcome is that clinical notes generated from conversation, i think information is lost