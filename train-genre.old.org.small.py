# Copyright (C) 2022-2023  Dmitri Roussinov
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
This code supports training a generator for on-topic synthetic texts  
Posted on https://github.com/dminus1/genre
For the the following paper:
BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff, EMNLP Findings, 2023
'''
import pandas as pd
import random


#re-write data with summaries, add genre level so can be used in a single model
labels = []
genre_label = "hyper"
labels.append(genre_label)
genre_label = "stories"
labels.append(genre_label)
genre_label = "A12"
labels.append(genre_label)
genre_label =  "arxiv"
labels.append(genre_label)
genre_label = "wiki"
labels.append(genre_label)
genre_label = "reviews"
labels.append(genre_label)
genre_label = "brown"
labels.append(genre_label)
genre_label = "stack"
labels.append(genre_label)
genre_label = "giga"
labels.append(genre_label)
genre_label = "legal"
labels.append(genre_label)
assert len(labels) == 10

#merge train and predictions to train reverse summarization with LDA keywords:
'''
Genres = open("all-genre.txt").readlines()
Genres = [g.replace('\n','') for g in Genres]
f = open("data/train_df-genre.tsv", "w", encoding='utf-8')
f.write("\ttarget_text\tinput_text\tprefix\n")
lines = []
for gi in range(10):
    genre = Genres[gi]
    path_pred = labels[gi] + "-1000-summarized.txt"
    path_test = labels[gi] + "-1000-to-summarize.tsv"
    #assert len(open(path_pred, encoding='utf-8').readlines()) == len(open(path_test, encoding='utf-8').readlines()) 
    for linep , linet  in zip(open(path_pred, encoding='utf-8').readlines(),open(path_test, encoding='utf-8').readlines()) :
        if len(linep) < 50: #header
            linep = linep
            #line = "\ttarget_text\tinput_text\tprefix\n"
        else:
            line = linet.split('\t')[0] + '\t' + linet.split('\t')[2] + '\t'  +  labels[gi] + ' ' + linep.split('\t')[2] + '\texpand\n' #no LDA version, just to train expansion
            #line = linet.split('\t')[0] + '\t' + linet.split('\t')[1] + '\t'  + linet.split('\t')[2] + ' sep ' + linep.split('\t')[2] + '\tsep\n'
            assert line.count('\t') == 3
            lines.append(line)
    path_test = path_test
random.shuffle(lines)
for line in lines:
    f.write(line)  # both topic words and low DF keywords
# f.write(linet.split('\t')[0] + '\t' + linet.split('\t')[2].replace('\t', '') + '\t' + linep.split('\t')[2].replace('\t', '') + '\texpand\n')  # both topic words and low DF keywords
quit() 
'''


from simpletransformers.t5 import T5Model


import argparse
parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--out', type=str, default='', help='optimization algorithm')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--model', type=str, default="t5-small", help='optimization algorithm')
parser.add_argument('--inp', type=str, default="data/train_df-genre.tsv", help='')
parser.add_argument('--batches', type=int, default=32, help='')
parser.add_argument('--size', type=int, default=256, help='')
parser.add_argument('--cut', type=int, default=4000, help='')

args = parser.parse_args()

#train_df = pd.read_csv("data/train_df.csv").astype(str) #
train_df = pd.read_csv(args.inp, sep="\t").astype(str) #
#train_df = pd.read_csv("data/train_df-genre.tsv", sep="\t").astype(str) #
#eval_df = pd.read_csv("data/eval_df.csv").astype(str) #
#eval_df = pd.read_csv("eval_df-genre.tsv", sep="\t").astype(str) #GITHUB

model_args = {
    "output_dir": args.out,  # 2022
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    #"max_seq_length":  256, #  trying to play with summarize:
    #"max_seq_length": 256, #, only as ablation
    #"max_seq_length": 256, #norm for good results with adding low DF keywords
    "max_seq_length": 512, #
    #"max_seq_length": 256, #did for most shuffled version
    #"max_seq_length": 128, # for keyword on topic generation   #for topic to keyword mapping, realized that topic is discreete, so ID is sufficient, but resulted in often too few good keywords, so re-running with 256
    "train_batch_size": 4   , 
    #"train_batch_size": 4   , #was in detached 2022
    #"train_batch_size": 1   , #, norm for 512 base
    #"train_batch_size":  1, #  #2 crashes with 512 even with "max_length": 1300 # mostly used 1 for 512, but smaller output can use more? 1  #6 too much for 256, 2 too much for 512
    #"train_batch_size": 8, # crashes with small and 512 len, OK for small with 256 len
    #"train_batch_size": 1, # > 1 does not work for 1024. 512 len
    #"train_batch_size": 4, #org, works fine with cats with max_seq_length": 128
    #"num_train_epochs": 100, #used for weeks+
    #"num_train_epochs": 4, #for overnight 2seg model training #
    "num_train_epochs": args.epochs,
    "save_eval_checkpoints": True,
    #"save_steps": 10, # worked, but slow?
    "save_steps": -1, #org
    "use_multiprocessing": False,
    "evaluate_during_training": False, #prg
    #"evaluate_during_training": True, # worked, but slow?
    #"evaluate_during_training_steps": 10,  worked, but slow?
    "evaluate_during_training_steps": 1500000, #org 
    #"evaluate_during_training_steps": 15000, #org
    "evaluate_during_training_verbose": False, # org
    #"evaluate_during_training_verbose": True,  worked, but slow?
    #"fp16": True, #
    "fp16": False,
    "learning_rate": 1e-3, #1e-3 default according to https://huggingface.co/transformers/main_classes/optimizer_schedules.html #  1e-4 #was used for geo
    #"wandb_project": "Question Generation with T5", # dis, fails asking for account
    "max_length": 4000,  #, not sure it affects anything?
    #"max_length": 1500,  # used 2 weeks+ for recent smaller experiments with training expansion,  may reduce memory requriments?
}



assert args.model == "t5-small" #Dec 2022
assert args.size ==  256 #Dec 2022

model = T5Model("t5", "t5-small", args=model_args, use_multiprocessing = False, use_multiprocessed_decoding = False) #
#model = T5Model("t5", "t5-large", args=model_args, use_multiprocessing = False, use_multiprocessed_decoding = False) # - does not start: "ValueError: Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection is on." ???
#model = T5Model(model_name = "outputs", args=model_args, model_type = 't5', use_multiprocessing= False, use_cuda= True) #was in detached in 2022 worked fine for Evalution and more
#model = T5Model("t5", "t5-large", args=model_args, use_multiprocessing = False, use_multiprocessed_decoding = False) #out of memory on desktop, even if small train batch and fp16
#model = T5Model("t5", "t5-small", args=model_args) #
#model = T5Model("t5", "t5-small", args=model_args, use_multiprocessing = False, use_cuda = False) #

#model = T5Model("t5", "t5-small", args=model_args, use_multiprocessing = False, use_multiprocessed_decoding = False, use_cuda = False)  worked, but slow?
#model = T5Model("t5", "t5-small", args=model_args, use_cuda = False, use_multiprocessing = False) #
#model = T5Model("t5", "t5-large", args=model_args)

model.train_model(train_df, eval_data=train_df) #GITHUB
#model.train_model(train_df, eval_data=eval_df)
print("Training Finished")
''' # dis
df = pd.read_csv("data/eval_df.tsv", sep="\t").astype(str)
#preds = model.predict(["ask_question: " + description for description in df["input_text"].tolist()])

#print (model.predict(["ask_question: what is the capital of Spain? "]))
preds = model.predict(["ask_question: what is the capital of Spain? "])
preds = preds
'''