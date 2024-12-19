
import os, json

print('CURRENT CWD: ', os.getcwd())

if False: # Check Keys
    file = open('/home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/mediNote-direct.jsonl') #gpt3-direct
    for i in file:
        print(json.loads(i).keys())
        break

#---- Evaluation ----
evaluationPath = '/home/willizhe/wz_stuff/evaluation/'

#NOTE: Make sure to delete saftey.pkl on restart
if False:
    from  utils.eval import *
    # inference_paths = "../data/inference"
    # output_path = "../data/combined_inferences.jsonl"
    # combined = combine(inference_paths, output_path)

    # gpt3_direct_eval_res = note_evaluation(
    #     # 'gpt3-direct',,
    #     'full_686_gpt3_CN_direct',
    #     save_path= evaluationPath + 'full_gpt3'
    # )

    # /home/willizhe/wz_stuff/LettiNote/MediNote/data/inference/mediNote_200_direct.jsonl
    meditron_7b_direct_trunc_eval_res = note_evaluation(
        # 'meditron-7b-direct', 
        'mediNote_200_direct', 
        save_path= evaluationPath +'mediNote_200_direct'
    )

if True: 
    # full_gpt3
    # old_686_full_gpt3_direct
    path = '/home/willizhe/wz_stuff/evaluation/mediNote_200_direct/'
    metrics = ['rouge', 'bleu', 'bert']
    # metrics = ['1', '2', 'L', 'Lsum']

    for metric in metrics:
        scorePath = path + metric + '.jsonl'

        if metric == 'rouge':
            sum1 = 0
            sum2 = 0
            sumL = 0
            sumLsum  = 0
            n = 0
            for row in open(scorePath):
                n += 1
                data = json.loads(row)
                sum1 += data[metric + '1']
                sum2 += data[metric + '2']
                sumL += data[metric + 'L']
                sumLsum += data[metric + 'Lsum']

            print(metric, '1: ', sum1/n)
            print(metric, '2: ', sum2/n)
            print(metric, 'L: ', sumL/n)
            print(metric, 'Lsum: ', sumLsum/n)

        else:
            total = 0
            n = 0
            for row in open(scorePath):
                data = json.loads(row)
                total += data[metric]
                n += 1
            print(metric, ': ', total/n)
            

            

        


    
