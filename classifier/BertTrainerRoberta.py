# Copyright (C) 2022-2023  Dmitri Roussinov
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
This code supports training and testing Roberta-based genre classifier  
Posted on https://github.com/dminus1/genre
For the the following paper:
BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff, EMNLP Findings, 2023
'''

import os
#import wandb
os.environ["WANDB_DISABLED"] = "true" #
#os.environ["WANDB_MODE"]="dryrun"

import torch
import torch.nn as nn
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AutoConfig #ROBERTA
#from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
import argparse  # VARIOUS
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_fscore_support  # AUC


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_label",
                        default="",
                        type=str,
                        required=False,
                        help="")
    parser.add_argument("--resume_from_checkpoint",
                        action='store_true',
                        help="")
    parser.add_argument("--auto_val",
                        action='store_true', #
                        help="")

    parser.add_argument("--shuf",
                        default=0, #
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--portion",
                        default=-1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--flat_lr",
                        action='store_true',
                        help="")
    parser.add_argument("--binary",
                        action='store_true',
                        help="")
    parser.add_argument("--eval_only",
                        action='store_true',
                        help="")
    parser.add_argument("--warmup_steps",
                        default=500,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seg_size",
                        default=512,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese, biobert.")
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--input_val",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--input_test",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--logging_steps",
                        default=100, #
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--per_device_train_batch_size",
                        default=1, #
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser
parser = setup_parser()
args = parser.parse_args()
assert  args.seg_size == 256 #ROBERTA
#device = 'cpu' #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(1)

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
model_name  = args.bert_model #ROBERTA
#model_name  = 'roberta-base' #ROBERTA
#model_name  = 'emilyalsentzer/Bio_ClinicalBERT'
#model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 256

tokenizer = RobertaTokenizerFast.from_pretrained(args.bert_model, do_lower_case=True)
#tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)


'''def read_20newsgroups(test_size=0.2):
    # download & load 20newsgroups dataset from sklearn's repos
    dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target
    # split into training & testing a return data as well as label names
    return train_test_split(documents, labels, test_size=test_size), dataset.target_names'''

import pandas as pd

import collections
MAX_DOC_LEN = 1000000 #ROBERTA
MAX_LINES = 1000000 #ROBERTA
Labels = collections.OrderedDict()
for  l in open("all-genre.txt").readlines():
    Labels[l.replace('\n','')] = len(Labels)

def FromGenreToGluon(inf, outf): #may be useful if takign data there
    f = open(outf, "w", encoding='utf-8')
    count = 0
    empty_count = 0
    lines = open(inf, encoding='utf-8').readlines()[1:] #TERENT
    for _ in range(args.shuf):
        random.shuffle(lines) 
    for line in lines: # 2022
    #for line in open(inf, encoding='utf-8').readlines()[1:]: # 2022
    #for line in open(inf, encoding='utf-8').readlines()[1:5000]: # 2022
    #for line in open(inf, encoding='utf-8').readlines()[:100]: #
        part = line.replace('\n', '').split('\t')
        text = part[2]
        if len(text) < 10:
            empty_count += 1
            text = 'dummy'
        text = text[-MAX_DOC_LEN:] #MASK
        f.write(part[0]+'\t' + str(Labels[part[1]]) + '\t*\t' + text + '\n')  # to Colaf.write('dummy_id\t'+str(int(part[0]=='True'))+'\t*\t' + part[1]+'\n') #to Cola
        count += 1
        if count > MAX_LINES:
            break
    f.close()
    print (f"empty inputs:{empty_count}")
    #assert empty_count < 100

def FromFastTextToGluon(inf, outf): #may be useful if takign data there
    f = open(outf, "w", encoding='utf-8')
    count = 0
    for line in open(inf, encoding='utf-8'): #
    #for line in open(inf, encoding='utf-8').readlines()[:100]: #
        part = line.replace('\n', '').split('\t')
        text = part[1]
        text = text[-MAX_DOC_LEN:] #MASK
        f.write('dummy_id\t' + part[0] + '\t*\t' + text + '\n')  # to Colaf.write('dummy_id\t'+str(int(part[0]=='True'))+'\t*\t' + part[1]+'\n') #to Cola
        count += 1
        if count > MAX_LINES:
            break
    f.close()

train_name = "BertTrainerGenre-tmp-train.txt"    # JULY 2022
test_name = "BertTrainerGenre-tmp-test.txt"  # JULY 2022
val_name =  "BertTrainerGenre-tmp-val.txt" #VALID
test_and_val_name =  "BertTrainerGenre-tmp-test-val.txt" #VALID

#train_name = "train-" + args.condition + "-full.tsv"  # SEG
#test_name = "test-" + args.condition + "-full.tsv"  # SEG

#fast_path = ""  # LAPTOP
#fast_path = "./fast-text/"  # LAPTOP #
#fast_path = "C:\\H\\work\\tutorials\\fast-text\\"  #on laptop
fast_path = "D:/C_backup/Users/xeb08186/work/tutorials/fast-text/" #on desktop
FromGenreToGluon(args.input, train_name ) # JULY 2022
FromGenreToGluon(args.input_test , test_name) # JULY 2022
TestLen = len(open(test_name, encoding='utf-8').readlines())-1 #VALID
FromGenreToGluon(args.input_val , val_name) #VALID
f = open(test_and_val_name, "w", encoding='utf-8')
f.write(''.join(open(test_name, encoding='utf-8').readlines() + open(val_name, encoding='utf-8').readlines()))
f.close()

#FromFastTextToGluon( fast_path + "train-" + args.condition + "-full-pos-under.txt", train_name) #was in JULY
#FromFastTextToGluon( "dummy-train.txt", train_name)
#FromFastTextToGluon( fast_path + "train-" + args.condition + "-full-over-sampling.txt", train_name)
#FromFastTextToGluon( "dummy-test.txt", test_name)
#FromFastTextToGluon( fast_path + "test-" + args.condition + "-full-no-sampling.txt", test_name)
#FromFastTextToGluon( fast_path + "test-" + args.condition + "-full-pos-under.txt", test_name) # test still undersampled to match one used for <= 500 steps saved
print("Format converted")
df = pd.read_csv(train_name, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
df_test = pd.read_csv(test_and_val_name, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence']) #VALID
#df_test = pd.read_csv(test_name, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence']) #norm

if args.portion != -1:
    start = args.portion * int(len(df.label.values) / 4) #splitting train into 4 parts if not fitting all to memory
    end = (args.portion + 1) * int(len(df.label.values) / 4)
    sentences_train = df.sentence.values[start:end]  # SPLIT
    labels_train = df.label.values[start:end]
else:
    sentences_train = df.sentence.values
    labels_train = df.label.values
sentences_test = df_test.sentence.values
labels_test = df_test.label.values

''' #ROBERTA dis
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df_test = pd.read_csv("./cola_public/raw/in_domain_test.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
print('Number of training sentences: {:,}\n'.format(df.shape[0]))
sentences_train = df.sentence.values
labels_train = df.label.values
sentences_test = df_test.sentence.values
labels_test = df_test.label.values
'''

train_texts = [e for e in sentences_train]
train_labels = [e for e in labels_train]
valid_texts = [e for e in sentences_test]
valid_labels = [e for e in labels_test]
# call the function
#(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

# tokenize the dataset, truncate when passed `max_length`,
# and pad with 0's when less than `max_length`
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        #item = {k: torch.tensor(v[idx]).to(device) for k, v in self.encodings.items()} #
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        #item["labels"] = torch.tensor([self.labels[idx]], dtype = torch.long).to(device) #
        item["labels"] = torch.tensor([self.labels[idx]], dtype = torch.long) #
        #item["labels"] = torch.tensor([self.labels[idx]])

        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

#was trying to increase dropouts but failed:
'''configuration = AutoConfig.from_pretrained(model_name) 
#configuration.hidden_dropout_prob = 0.5
#configuration.attention_probs_dropout_prob = 0.5
model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, config=configuration)'''

# load the model and pass to CUDA

model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(Labels)) #ROBERTA
#model.dropout = nn.Dropout(.0) # does not affect

#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) #
#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device) #
#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to(device)
#model.to(device) #

model_label = '-l' if args.bert_model == "bert-large-uncased" else '-s' #ROBERTA
ResulFileName = os.path.basename(__file__)  + ".lr" + str(args.learning_rate) + ".b" + str(args.gradient_accumulation_steps)  + ".data.txt"
already_reported = False
eval_count = 0
test_roc_best = 0
not_increased = 0
vf1_max = 0.
f1_max = 0.


from sklearn.metrics import accuracy_score

#ROBERTA
def compute_metrics(pred):
  global eval_count
  global test_roc_best
  global not_increased
  #eval_count = 0
  eval_count += 1
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels[:TestLen], preds[:TestLen]) # undis
  vacc = accuracy_score(labels[TestLen:], preds[TestLen:]) #VALID

  #assert len(preds[TestLen:]) > 0
  #vacc = accuracy_score(labels[TestLen:], preds[TestLen:]) #SPLIT

  pre, rec, f1, support = precision_recall_fscore_support(labels[:TestLen], preds[:TestLen], average='weighted')  #from cats project # JULY 2022 undis
  vpre, vrec, vf1, support = precision_recall_fscore_support(labels[TestLen:], preds[TestLen:], average='weighted')  #VALID
  global vf1_max
  global f1_max
  if args.auto_val:
      if vf1 > vf1_max:
          vf1_max = vf1
          f1_max = f1
  #vpre, vrec, vf1, support = precision_recall_fscore_support(labels[TestLen:], preds[TestLen:], average='weighted')  #from cats project #SPLIT # JULY 2022 undis
  if False:
  #if args.confuse:
   print(confusion_matrix(labels[:TestLen], preds[:TestLen],labels=[i for i in range(10)]))  #
  #print(confusion_matrix(labels[:TestLen], preds[:TestLen],labels=["0","1","2","3","4","5","6","7","8","9"]))  #
  #print(confusion_matrix(labels[:TestLen], preds[:TestLen]))  #
  #print(confusion_matrix(labels[:TestLen], preds[:TestLen], labels=[l.replace('\n', '') for l in open("../simple-transformers/genre/all-genre.txt").readlines()]))  #

  #if True: #
  if args.binary:
      assert False #VALID
      scores = np.zeros(len(labels)) #
      for i in range (len(labels)):
          sum = 0
          for j in range(len(pred.predictions[i])):
              sum += np.exp(min(10, pred.predictions[i][j])) #
              #sum += np.exp(pred.predictions[i][j])
          if sum > 1e-10: #
           scores[i] = np.exp(min(10, pred.predictions[i][pred.predictions[i].argmax()]))/sum #
           #scores[i] = np.exp(pred.predictions[i][pred.predictions[i].argmax()])/sum
          else:
           scores[i] = 0 #
      # scores = [np.exp(e) / np.exp(e).sum() for e in pred.predictions]
      #
      scores = [np.exp(e) / np.exp(e).sum() for e in pred.predictions]
      auc  = roc_auc_score([l[0]  for l in labels[:TestLen]], [s[1] for s in scores[:TestLen]]) #GEN, only binary so far # JULY 2022 dis

  '''f = open("run.txt",  "w",encoding='utf-8') #
  #f = open("run.txt",  "w",encoding='utf-8')
  for e in scores:
      f.write(str(e) + '\n')
  f.close()'''


  #vauc = roc_auc_score(labels[TestLen:],  scores[TestLen:], multi_class = "ovr", average = 'weighted')
  #avp = average_precision_score(labels, scores, average='weighted')

  fd = open(ResulFileName, "a+")
  report_str = str(f1) + '\n' + str(vf1) + '\n'  #VALID
  #report_str = str(f1) + '\n'  # JULY 2022
  #report_str = str(auc) + '\n'  # later can add test cost
  #report_str = str(auc) + '\t' + str(vauc) + '\n'  # later can add test cost #SPLIT
  #report_str = str(count_results) + '\t' + str(auc) + '\t' + str(vauc) + '\n'  # later can add test cost
  #count_results += 1
  fd.write(report_str)
  fd.close()

  '''if not (auc > test_roc_best + .003): #changed only after readm started # JULY 2022 dis
      not_increased += 1
  else:
      not_increased = 0
  test_roc_best = max(auc, test_roc_best)  # '''
  '''if not_increased >= 5:  #
      # if not_increased >= 4:  #  dis
      # if test_roc <= test_roc_best - .02 and gcount > EachSave*10 and test_roc_best > .55: # dis
      print("slow progress, terminating...")
      sys.exit()''' # dis

  return {'accuracy': acc, 'f1': f1, 'auc': auc} if args.binary else {'accuracy': acc, 'f1': f1,'vaccuracy': vacc, 'vf1': vf1} #VALID
  #return {'accuracy': acc,'f1': f1, 'auc': auc} if args.binary else {'accuracy': acc,'f1': f1}
  #return {'accuracy': acc,'f1': f1, 'auc': auc} #

'''def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }'''

training_args = TrainingArguments(
    #with ChatGPT help, able to disable saving, but empty folders still created
    #ChatGPT: Unfortunately, there is no direct option to disable both saving and evaluation in the current version of the library.
    save_strategy='no', 
    #evaluation_strategy='no', #errs
    evaluation_strategy="steps",     # evaluate each `logging_steps` #norm #
    eval_steps = args.logging_steps, #ROBERTA
    #load_best_model_at_end='no', #errs
    #load_best_model_at_end=False,     
    #save_total_limit = 0 , #still saves for some reason
    #output_dir=None, # to skip saving, according to ChatGPT, but errs
    output_dir=args.output_dir,  # ROBEERTA was './results',  # output directory #norm #ChaGPT: required, so folder created even if left empty
    resume_from_checkpoint=args.resume_from_checkpoint,  #ROBERTA
    #resume_from_checkpoint = 'results\checkpoint-500', #worked
    learning_rate = args.learning_rate, ##ROBERTA was 1e-06
    gradient_accumulation_steps = args.gradient_accumulation_steps,  # was 8, #ROBERTA
    per_device_train_batch_size= args.per_device_train_batch_size,  #ROBERTA,was 5 prior
    num_train_epochs=args.num_train_epochs,              #
    #per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size= 8, #ROBERTA ok for Roberta Large, could be even more?
    #per_device_eval_batch_size= args.per_device_train_batch_size, #ROBERTA, was 1
    #per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler  - DR: this does not affect "multiples" error
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=args.logging_steps,               #ROBERTA
    #logging_steps=100,               #
    #logging_steps=500,               # log & save weights each logging_steps
)

if args.flat_lr:
    training_args.lr_scheduler_type = 'constant' #ROBERTA

trainer = Trainer(
    #place_model_on_device = "cpu", #DR: does not recog
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

# train the model
trainer.train()

if args.auto_val:
    print("\nmax val:", vf1_max, "max test:", f1_max,"\n")
    open(f"log{args.log_label}.txt",'a').write(f"\n{args},{f1_max}") 
    #open('log.txt','a').write(f"\n{args},{f1_max}") 
