from utils.infer import *
from utils.data import *
from utils.chat import *
from utils.generate import *

modelName = "gpt3"
mode = "direct-gpt" # :param mode: str, mode of inference (direct-gpt, generator-gpt)
max_tokens = 20000
num_samples = 686
inputPath = "LettiNote/MediNote/data/misc/full_similarNote.jsonl" 
outputPath = 'LettiNote/MediNote/data/inference/full_686_gpt3_CN_1shot.jsonl' 

# Generate clinical notes from conversations using an OpenAI model.
infer_openai(
    input_path=inputPath,
    output_path=outputPath,
    model_name=modelName,
    mode=mode,
    max_tokens=max_tokens,
    shots=1,
    num_samples=num_samples
)

#NOTE: outcome is that clinical notes generated from conversation, i think information is lost