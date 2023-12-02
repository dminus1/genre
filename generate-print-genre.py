# Copyright (C) 2022-2023  Dmitri Roussinov
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
This code supports generating on-topic synthetic texts based on trained models.
Posted on https://github.com/dminus1/genre
For the the following paper:
BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff, EMNLP Findings, 2023
'''

from simpletransformers.t5 import T5Model
import pandas as pd
from pprint import pprint
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, average_precision_score, precision_recall_fscore_support #
import random
#seed_id = 9 #
import sys


import argparse
parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--model', type=str, default='outputs', help='')
parser.add_argument('--out', type=str, default='predictions.txt', help='')
parser.add_argument('--inp', type=str, default='data/eval_df-print-genre.tsv', help='')
parser.add_argument('--batches', type=int, default=4, help='')
parser.add_argument('--seed', type=int, default=9, help='')
parser.add_argument('--cap', type=int, default=10000, help='')
parser.add_argument('--top_k', type=int, default=50, help='')
parser.add_argument('--top_p', type=int, default=95, help='')
parser.add_argument('--maxl', type=int, default=5000, help='')




args = parser.parse_args()
random.seed(args.seed)

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    #"max_seq_length": 70, # for cats, to match train
    #"max_seq_length":  128, #  dis: reading from model for keywords (but in JSON it says 256 ??), if accidetally used on shuffled versions, produces poor results?
    #"max_seq_length":  256, #
    #"max_seq_length":  512, #on St eve run models trained for 256, matters?
    #"eval_batch_size": 16, # 16 was fine, but cut off ??? - NO! it was use of double quotes " that messes tsv files up! #24 ok too but too risky for overnight        , # 16 ok base wih max_length 500; for  #2,  # , 8 ok for 512 base, but  crashed once,   32 crashes on 256,
    #"eval_batch_size": 16,  # too much for 256 base
    #"eval_batch_size": 1,  # norm for low DF tests (256 base), could be more for small?
    #"eval_batch_size": 1, # only when trying in parallel with train  since in parallel
    #"eval_batch_size": 8,  #fails for 256 (base?)
    #"eval_batch_size": 16,  # trying  for 512 small, worked once, but failed another
    #"eval_batch_size": 12,  # trying  for 512 small - wanted 1 hr for brown
    #"eval_batch_size": 6,  # ok  for 512 small
    #"eval_batch_size": 2,  #norm for 512 base. #
    "eval_batch_size": args.batches,#   #norm and  fastest for 512 small, but OK for 512 base? - No
    #"eval_batch_size": 4,#   #norm and  fastest for 512 small, but OK for 512 base? - No
    "save_eval_checkpoints": False,
    "use_multiprocessing": False, #norm
    "num_beams": 1,
    #"num_beams": None, #norm
    "do_sample": True,
    ##assuming of output in chars, to match inputs:
    #"max_length": 1500, #  for 2 segments experiments
    #"max_length": 500, # was for summaries only
    #"max_length": 1300, #  for summaries only
    #"max_length": 500, #  for summaries only
    "max_length": args.maxl, # 2022 to match training + 1k for stat deviations
    #"max_length": 5000, # 2022 to match training + 1k for stat deviations
    #"max_length": 1000, #  to match training
    #"max_length": 4000, # to match models with :4000
    #"max_length": 5000, # afects anything ??? to match data generating code
    #"max_length": 600, # resul will be longer?
    #"max_length": 50, #for genre classification
    #"max_length": 300, # to match max input len in chars
    #"max_length": 1000, #used for most older  tests with low DF back
    #"max_length": 50,
    #"top_k": 10, # - slow
    #"top_k": 50, #norm
    "top_k": args.top_k,
    #"top_p": 0.97,
    #"top_p": 0.95, #norm
    "top_p": args.top_p/100.,
    #"num_return_sequences": 1, # i check only one anyway - but does not work ??
    "num_return_sequences": 2, # norm
    #"num_return_sequences": 3,
} # in the one worked, some setting where changed, see "generate that run.py"

model = T5Model(model_name = args.model, args=model_args, model_type = 't5', use_multiprocessing= False, use_cuda= True) #FRECENT
#model = T5Model(model_name = "outputs", args=model_args, model_type = 't5', use_multiprocessing= False, use_cuda= True) #norm
#CARE: have to read a model from disk along with JSON so to turn off multiprocessing, otherwise crashes, tried:
#model = T5Model("t5", "t5-base", args=model_args, use_multiprocessing = False, use_multiprocessed_decoding = False) assert False
#model = T5Model("t5", "t5-small", args=model_args, use_multiprocessing = False, use_multiprocessed_decoding = False, use_cuda= False) assert False
#model = T5Model(model_name = "outputs\best_model", args=model_args, model_type = 't5', use_multiprocessing= False, use_cuda= False) # can't read model

while True:
    df = pd.read_csv(args.inp, sep="\t").astype(str) #
    #df = pd.read_csv("data/eval_df-print-genre.tsv", sep="\t").astype(str) #
    #df = pd.read_csv("data/eval_df-print.tsv", sep="\t").astype(str) #

    #df = pd.read_csv("data/eval_df.tsv", sep="\t").astype(str)
    #freeze_support() #
    #CARE: crucial to keep reading texts from the test file, since does not work when the text supplied directly

    #[ assert(len(description) > 100) for description in df["input_text"].tolist()]

    for description in df["input_text"].tolist():
        if len(description) < 50:
            description = description
        #assert len(description) > 50

    #prefix = "sep" #
    #prefix = "rep" # #realized used for last few days, likely not matching those in train!
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
    #movies = open("movies.txt").read()
    #movies = "Cameron is a director.\nTom Cruz is an actor.\nTarantino is a director.\nHitchcock is a producer.\nKubrick is a producer.\nScorsese is a producer.\nDamon is a actor.\nDe Niro is a actor.\nPitt is a actor.\nKubrick is  a director.\nNolan is a director.\nClooney is an actor.\nAffleck is an actor.\n"
    #capitals = open("capitals.txt").read()
    #capitals = "Q: What is the capital of France? A: Paris\n\nA:What is the capital of UK? A: London\n\nQ:What is the capital of Russia? A: Moscow\n\nQ: What is the capital of Spain? A: Madrid\n\nQ: What is the capital of China? A: Beijing\n\nQ: What is the capital of Estonia? A: Tallinn\n\n"
    #capitals = "Paris is the capital of France.\n\nLondon is  the capital of UK.\n\nMoscow is the capital of Russia.\n\nMadrid is the capital of Spain.\n\nBeijing is the capital of China.\n\nTallinn is the capital of Estonia.\n\nTokyo is the capital of Japan.\n\nSeoul is the capital of Korea.\n\nKiev is the capital of Ukraine.\n\nWashington is the capital of US.\n\nVilnius is the capital of Lithuania.\n\nMinsk is the capital of Belarus.\n\nWarsaw is the capital of Poland.\n\n"
    #capitals = ["Paris is the capital of France. London is  the capital of UK. Moscow is the capital of Russia. Madrid is the capital of Spain. Beijing is the capital of China. Tallinn is the capital of Estonia. Tokyo is the capital of Japan. Seoul is the capital of Korea. Kiev is the capital of Ukraine. Washington is the capital of US. Vilnius is the capital of Lithuania. Minsk is the capital of Belarus. Warsaw is the capital of Poland."]
    #capitals = ["Paris is the capital of France.\nLondon is  the capital of UK.\nMoscow is the capital of Russia.\nMadrid is the capital of Spain.\nBeijing is the capital of China.\nTallinn is the capital of Estonia.\nTokyo is the capital of Japan.\nSeoul is the capital of Korea.\nKiev is the capital of Ukraine.\nWashington is the capital of US.\nVilnius is the capital of Lithuania.\nMinsk is the capital of Belarus.\nWarsaw is the capital of Poland.\n"]
    #print([movies, capitals])
    #preds = model.predict([movies, capitals])
    #preds = model.predict(capitals)  #
    preds = model.predict(["expand: " + description for description in df["input_text"].tolist()[:args.cap]]) #
    #preds = model.predict(["expand: " + description for description in df["input_text"].tolist()]) ##norm # 2022 undis
    #preds = model.predict(["summarize: " + description for description in df["input_text"].tolist()]) #
    #preds = model.predict(["sep: " + description for description in df["input_text"].tolist()]) #
    #preds = model.predict(["genre: " + description for description in df["input_text"].tolist()]) #
    #preds = model.predict(["ask_question:" + description for description in df["input_text"].tolist()]) #
    #preds = model.predict(["ask_question: " + description for description in df["input_text"].tolist()]) #
    #preds = model.predict(["relations: " + description for description in df["input_text"].tolist()])
    f = open(args.out, "w", encoding='utf-8') #
    #f = open("predictions.txt", "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n") #
    for i in range(len(preds)):
    #for p in preds:
        p = preds[i][0] #
        #p = preds[i]
        #f.write(str(p).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')  + '\n') #to send to SS
        f.write(str(i) + '\t' + 'dummy' + '\t' + str(p).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')  + '\tsummarize\n')
        #f.write(str(i) + '\t' + '__id__A4-brown-bnc-cen.ol' + '\t' + str(p).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')  + '\tsummarize\n') # removed extra ' ' on both sides of 'dummy'
        #f.write(str(i) + '\t' + ' dummy ' + '\t' + str(p).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')  + '\tsummarize\n') #to generate keyword for follow up generations##
        #f.write(str(i+1) + ' ' +  str(p)+'\n')
        #f.write(str(i+1) + ' ' +  str(p)+'\n')
        #f.write(str(p)+'\n')
    f.close() # dis since failed on K&H with character map ??
    f = f
    print(str(p)) #GENRE
    pred_label =  [p[0] for p in preds]

    '''  DIS, works but slow
    assert len(df['target_text']) == len(preds)
    pred_label =  [p[0] for p in preds]
    '''
    # pred_label = ['random' if random.randint(0,100) == 1 else p for p in df['target_text'] ] #verified drops ~1% indeed

    #pred_label =  ['IsA' for _ in df['target_text']]
    #pred_label = df['target_text'] #indeed give 100%-s

    #''' #
    test_results = {} #   DIS, works but slow
    test_acc = accuracy_score(df['target_text'], pred_label)
    test_results['acc'] = test_acc
    pre, rec, f1, support = precision_recall_fscore_support(df['target_text'], pred_label, average='weighted')  #
    # pre, rec, f1, support = precision_recall_fscore_support(test_csv['label'], pred_label, average='binary') #norm for binary tasks
    test_results['f1'] = f1
    test_results['rec'] = rec
    test_results['pre'] = pre
    results = {"test": test_results}  # DIS, works but slow
    print(results)
    #print (confusion_matrix(df['target_text'], pred_label, labels = [str(i) for i in range(25)]))

    '''
    questions = df["target_text"].tolist()
    
    with open("test_outputs_large/generated_questions_sampling.txt", "w") as f:
        for i, desc in enumerate(df["input_text"].tolist()):
            pprint(desc)
            pprint(preds[i])
            print()
    
            f.write(str(desc) + "\n\n")
    
            f.write("Real question:\n")
            f.write(questions[i] + "\n\n")
    
            f.write("Generated questions:\n")
            for pred in preds[i]:
                f.write(str(pred) + "\n")
            f.write("________________________________________________________________________________\n")
    '''
    break  # GENRE
