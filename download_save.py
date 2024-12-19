# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("AGBonnet/medinote-7b")
# model = AutoModelForCausalLM.from_pretrained("AGBonnet/medinote-7b")

# /scratch/jjcorso_root/jjcorso98/willizhe/model/medinote-7b
# model.save_pretrained('/home/willizhe/wz_stuff/mediModel')


# Load model directly
tokenizer = AutoTokenizer.from_pretrained("AGBonnet/medinote-7b")
model = AutoModelForCausalLM.from_pretrained("AGBonnet/medinote-7b")

print(type(model))
# /scratch/jjcorso_root/jjcorso98/willizhe/model/medinote-7b
# model.save_pretrained('/scratch/jjcorso_root/jjcorso98/willizhe/model')