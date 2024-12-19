"""
The following code is direcetly pulled from the eval_copy.ipynb file.
Any changes from the original eval.ipynb file is marked with "#myadd"

"""
from utils.infer import *
import pandas as pd
from  utils.eval import *
from utils.chat import *
from utils.data import *

# jupyter specific
# %reload_ext autoreload
# %autoreload 2

# [markdown]
# ### Combining inferences


inference_paths = "../data/inference"
output_path = "../data/combined_inferences.jsonl"

combined = combine(inference_paths, output_path)

# [markdown]
# ### Building evaluation inputs


all_models = MODELS_TO_MODE.keys()


build_evaluation_inputs(output_path, models = all_models)

# [markdown]
# ### # [OLD] Evaluation GPT 3 direct


gpt_3_eval_res = note_evaluation('direct-gpt')


#display(gpt_3_eval_res.sort_values(by='aggregated_score', ascending=False)) ipynbcode

# [markdown]
# ### Evaluation meditron-7b-summarizer


summarizer7_eval_by_sample , summarizer7_eval_by_key = summary_evaluation('meditron-7b-summarizer')

# [markdown]
# ### Evaluation meditron-13b-summarizer


summarizer13_eval_by_sample , summarizer13_eval_by_key = summary_evaluation('meditron-13b-summarizer')

# [markdown]
# ### Merging summaries eval (and computing aggragated scores)

#NOTE: COMMENDTED OUT BC summarizer_models NOT DEFINED
#final_summary_eval = merge_summary_evaluation(summarizer_models) 

#display(final_summary_eval) ipynb code


evaluation_path = 'evaluation'
eval_by_sample_path = '_eval_res/summary_eval_by_sample.jsonl'
eval_by_key_path = '_eval_res/summary_eval_by_key.jsonl'

summarizer7_eval_by_sample = load_file(f"{evaluation_path}/meditron-7b-summarizer{eval_by_sample_path}")
summarizer7_eval_by_key = load_file(f"{evaluation_path}/meditron-7b-summarizer{eval_by_key_path}")
summarizer13_eval_by_sample = load_file(f"{evaluation_path}/meditron-13b-summarizer{eval_by_sample_path}")
summarizer13_eval_by_key = load_file(f"{evaluation_path}/meditron-13b-summarizer{eval_by_key_path}")

#---IPYNB code---
# display(summarizer7_eval_by_sample.sort_values(by='aggregated_score', ascending=False))
# display(summarizer7_eval_by_key)
# display(summarizer13_eval_by_sample.sort_values(by='aggregated_score', ascending=False))
# display(summarizer13_eval_by_key)
#---------------


# [markdown]
# ### Evaluation meditron-7b-direct-trunc


meditron_7b_direct_trunc_eval_res = note_evaluation('meditron-7b-direct-trunc')

# [markdown]
# ### Evaluation meditron-13b-direct-trunc


meditron_13b_direct_trunc_eval_res = note_evaluation('meditron-13b-direct-trunc')

# [markdown]
# ### Evaluation meditron-7b-generator


meditron_7b_generator_eval_res = note_evaluation('meditron-7b-generator')

# [markdown]
# ### Evaluation meditron-13b-generator


meditron_13b_generator_eval_res = note_evaluation('meditron-13b-generator')

# [markdown]
# ### Evaluation gpt3-direct


gpt3_direct_eval_res = note_evaluation('gpt3-direct')

# [markdown]
# ### Evaluation mistral-7b-direct


mistral_7_b_eval_res = note_evaluation('mistral-7b-direct')

# [markdown]
# ### Evaluation gpt3-generator-gpt


gpt3_generator_gpt_eval_res = note_evaluation('gpt3-generator-gpt')

# [markdown]
# ### Evaluation gpt3-generator-7b


gpt3_generator_7b = note_evaluation('gpt3-generator-7b')

# [markdown]
# ### Evaluation gpt3-generator-13b


gpt3_generator_13b_eval_res = note_evaluation('gpt3-generator-13b')

# [markdown]
# ### Evaluation  meditron-7b-generator-gold


meditron_7b_generator_gold_eval_res = note_evaluation('meditron-7b-generator-gold')

# [markdown]
# ### Evaluation meditron-13b-generator-gold


meditron_13b_generator_gold_eval_res = note_evaluation('meditron-13b-generator-gold')

# [markdown]
# ### Evaluation gpt4-direct


gpt4_direct_eval_res = note_evaluation('gpt4-direct')

# [markdown]
# ### Evaluation llama-2-7b-chat


llama_2_7b_chat_eval_res = note_evaluation('llama-2-7b-direct')

# [markdown]
# ### Evaluation llama-2-13b-chat


llama_2_13b_chat_eval_res = note_evaluation('llama-2-13b-direct')

# [markdown]
# ### Merging note evaluation


final_eval = merge_note_evaluations(model_names= NOTE_MODELS)


#display(final_eval) ipynbcode


note_eval_res = {}
evaluation_path = 'evaluation'
for model in NOTE_MODELS:
    note_eval_res[model] = load_file(f"{evaluation_path}/{model}_eval_res/all_scores.jsonl")
    print(f"\n\n{model}\n")
    #display(note_eval_res[model].sort_values(by='aggregated_score', ascending=False)) ipynb code

# [markdown]
# ### Elo ranking


elo_rankings = elo_ranking(path='evaluation/elo_inputs.jsonl')


#display(elo_rankings.sort_values(by='final_score', ascending=False)) ipynbcode


for score_historie in elo_rankings['score_histories']:
    plt.plot(score_historie)

plt.xlabel('Number of games')
plt.ylabel('Elo score')
plt.title('Elo score histories')
plt.legend(elo_rankings['model'])
plt.show()


