# Copyright (C) 2022-2023  Dmitri Roussinov
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
This code supports training and testing Bert-based genre classifier  
Posted on https://github.com/dminus1/genre
For the the following paper:
BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff, EMNLP Findings, 2023
'''

import os
import sys
# import wandb
os.environ["WANDB_DISABLED"] = "true"  #
# os.environ["WANDB_MODE"]="dryrun"

import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
#from sklearn.datasets import fetch_20newsgroups #LAPTOP
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import argparse  # VARIOUS
from torch.nn import CrossEntropyLoss, BCELoss #FORWARD
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_recall_fscore_support  # AUC

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuf",
                        default=0, #
                        type=int,
                        help="Total number of training epochs to perform.") 
    parser.add_argument("--log_label",
                        default="",
                        type=str,
                        required=False,
                        help="")
    parser.add_argument("--flat_lr",
                        action='store_true',
                        help="")
    parser.add_argument("--resume_from_checkpoint",
                        action='store_true',
                        help="")
    parser.add_argument("--auto_val",
                        action='store_true',
                        help="")
    parser.add_argument("--eval_only",
                        action='store_true',
                        help="")
    parser.add_argument("--data_only",
                        action='store_true',
                        help="")

    parser.add_argument("--warmup_steps",
                        default=500,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--binary",
                        action='store_true',
                        help="")
    parser.add_argument("--frieze_embeds",
                        action='store_true',
                        help="")
    #parser.add_argument("--condition",
    #                    default=None,
    #                    type=str)

    parser.add_argument("--portion",
                        default=-1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--unfrozen_segs",
                        default=4,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--layer_frozen",
                        default=8,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--layer",
                        default=12,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seg_size",
                        default=512,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
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
    parser.add_argument("--initializer_range", #INIT
                        default= .02, #5e-5,
                        type=float,
                        help="")
    return parser


parser = setup_parser()
args = parser.parse_args()


SEG_SIZE = args.seg_size # JULY 2022
#SEG_SIZE = 256 # JULY 2022
#SEG_SIZE = 16 # 32 mostly used #norm for Medical so far, but more would be better # 512 # 128 # 64 # 512 #  512 # norm for BERT
NUM_BERT_SIZE_SEGMEENTS  = .5 # JULY 2022
#NUM_BERT_SIZE_SEGMEENTS  = 2 # # 14 # to take from the doc, 10 for close to full text Medical, but still not for all
#NUM_SEGMENTS_FOR_DOCS = 4 #temp for quick test
NUM_SEGMENTS_FOR_DOCS = int(NUM_BERT_SIZE_SEGMEENTS * 512/SEG_SIZE)  #for Medical
CONST_NUM_SEQUENCES =  NUM_SEGMENTS_FOR_DOCS #
MAX_DOC_LEN = 1000000 #NUM_BERT_SIZE_SEGMEENTS*512*7 #, estimate assumes chars/token average, to speed up 1000000 # in characters
MAX_LINES = 1000000

assert SEG_SIZE <= 512


# device = 'cpu' #
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
model_name = args.bert_model  # VARIOUS
# model_name  = 'emilyalsentzer/Bio_ClinicalBERT'
# model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
#max_length = 512

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)


import pandas as pd

import collections
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

train_texts = [e for e in sentences_train]#[:100] #
train_labels = [e for e in labels_train]#[:100] #
valid_texts = [e for e in sentences_test]#[:100] # #label 'valid' inherited from tutorial for the test set
valid_labels = [e for e in labels_test]#[:100] #
# call the function
# (train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

Tail = False # July 2022
#Tail = True # presuming it is overall better?

def make_features_from_ids(tokens, CONST_NUM_SEQUENCES_): #CARE: taken from A P code, but adapted for HuggingFace tokenizer output format (no min length specified)
    #tokens = [101] + tokens[1:9] + [102] #
    assert len(tokens) > 1 #
    #truncates  where padding starts if any
    assert tokens[0] != 0
    for pad_start in range(len(tokens)):
        if tokens[pad_start] == 0 :
            tokens = tokens[:pad_start]
            break
    assert tokens[0] == 101 and tokens[-1] == 102 and not (0 in tokens[1:-1])
    tokens = tokens[1:-1] #strip off [CLS] and [SEP] LAPTOP
    assert not (101 in tokens) and not (102 in tokens)

    SegTextLen = SEG_SIZE - 2
    num_sections = CONST_NUM_SEQUENCES_ #INCOMPLETE, now assume our segmenting still too short to need padding segs #int((len(tokens)  - 2 + SEG_SIZE - 2 + 1) / SegTextLen) #SEG -2 since [CLS] and [SEP] already added
    #assert  CONST_NUM_SEQUENCES_ == 2 #have not checked others except 10, should not be any remainder
    #assert num_sections*SegTextLen + 2 ==  len(tokens) # dis for shortened inputs - they may fail for inputs shorter than in the segs?
    #assert (len(tokens)  - 2)  %  (SegTextLen) == 0 # dis for shortened inputs - they may fail for inputs shorter than in the segs?
    sections = []
    masks  = []
    #sections_deb = []
    if Tail:
        start_t_raw = len(tokens) - 1 - (SEG_SIZE-2) * CONST_NUM_SEQUENCES_ + 1  #  CARE: need to add 1 otherwise last char lost, was BUG before Sep 2021
        start_t = max(0, start_t_raw)
    else:
        start_t = 0
    assert num_sections == CONST_NUM_SEQUENCES_ #SEG
    assert 101 not in tokens and 102 not in tokens
    #start_t = max(0, start_t)  #SEG +1 since skipping first [CLS] #TAIL dis, not needed for Head, and wrong for tail
    num_sections_use = min(num_sections, CONST_NUM_SEQUENCES_)
    sections_deb = []
    for t in range(num_sections_use):
        this_section = tokens[start_t + t * SegTextLen:start_t + (t + 1) * SegTextLen]  #works same for Tail and Head
        sections_deb.append(this_section)
        #sections_deb.append(this_section)
        input_ids = [101] + this_section + [102] + [0] * (SEG_SIZE - len(this_section) - 2) #can't rely on Hugging tokenizer padding, since it does not always do it to the length asked
        assert len(input_ids) == SEG_SIZE
        #input_ids, _, _, _ = convert_to_features(this_section, tokenizer)
        sections.append(input_ids)
        mask  = [1]*(len(this_section)+2) + [0] * (SEG_SIZE - len(this_section) - 2)
        masks.append(mask)
        if mask[-1] == 0:
            mask = mask
        assert len(mask) == len(input_ids)
    if Tail:
        assert sum(sections_deb, [])  == tokens[-SegTextLen*CONST_NUM_SEQUENCES_:]
    else:
        assert sum(sections_deb, []) == tokens  # truncation already done when tokenizier called
    '''while len(sections) < CONST_NUM_SEQUENCES_:  #LAPTOP dis

        this_section = ["[unused1]"] #CARE: there is nounused0! it would be mapped to [UNK], which is 100
        input_ids, input_mask, segment_ids, _ = convert_to_features(this_section, tokenizer) #
        sections.append(input_ids)'''
    return sections , masks #
    #return sections #, masks

def Segmentize(ids, tokens, masks, labels, val = False): #merges snippet lines for the same pair
    out = []; tout = []; mout = []; lout = []
    for doc, label in zip(ids, labels):

        segs_flat = []
        masks_flat = []
        #doc = doc[:10] #errs later since [SEP] cut off
        segs, masks = make_features_from_ids(doc, NUM_SEGMENTS_FOR_DOCS) #
        #segs = make_features_from_ids(doc, NUM_SEGMENTS_FOR_DOCS)
        for s in segs: #can do sum(segs, []) instead?
          for id in s:
            segs_flat.append(id)
        for m in masks:  # can do sum(segs, []) instead?
                for m1 in m:
                    masks_flat.append(m1)
        out.append(segs_flat)
        tout.append(segs_flat) #those are not used anyway, only to avoid reverse-eng NewsGroupsDataset() taken from a template
        mout.append(masks_flat)
        #mout.append(segs_flat)
        lout.append(label)

    assert  len(out) and len(tout) and len(mout) and len(lout)    #  JULY 2022
    return out, tout, mout, lout

Synthetic = False #
VocSize = 100
assert VocSize < 30000-100
NumDocs = 100  # only for Synthetic
def Gen(ids, tokens, masks, labels, val=False):  # merges snippet lines for the same pair
        out = [];
        tout = [];
        mout = [];
        lout = []
        for _ in range(NumDocs):

            segs_flat = []
            count = 0
            sum = 0
            for _ in range(NUM_SEGMENTS_FOR_DOCS):
                ids = []
                for _ in range(SEG_SIZE-2):
                    count += 1
                    id = random.randint(0, VocSize)
                    sum += id
                    ids.append(101 + id)

                seg = [101] + ids + [102]
                segs_flat = segs_flat + seg

            out.append(segs_flat)
            tout.append(segs_flat)
            mout.append(segs_flat)
            label = sum / count > 1/2 * VocSize
            lout.append(label)

        return out, tout, mout, lout


if Synthetic:
    train_encodings =  tokenizer([' '.join(['A' for _ in range(10)]) for _ in range(NumDocs) ], truncation=True, padding=True, max_length=int((SEG_SIZE-2)*NUM_SEGMENTS_FOR_DOCS+2))
    valid_encodings =  tokenizer([' '.join(['A' for _ in range(10)]) for _ in range(NumDocs) ], truncation=True, padding=True, max_length=int((SEG_SIZE-2)*NUM_SEGMENTS_FOR_DOCS+2))
    train_encodings["input_ids"], train_encodings["token_type_ids"], train_encodings[
        "attention_mask"], train_labels = Gen(train_encodings["input_ids"], train_encodings["token_type_ids"],
                                                     train_encodings["attention_mask"], train_labels)
    valid_encodings["input_ids"], valid_encodings["token_type_ids"], valid_encodings[
        "attention_mask"], valid_labels = Gen(valid_encodings["input_ids"], valid_encodings["token_type_ids"],
                                                     valid_encodings["attention_mask"], valid_labels, val=True)
else:
    if Tail:
        assert False #VALID
        train_encodings = tokenizer(train_texts, truncation=False, padding=False) #
        valid_encodings = tokenizer(valid_texts, truncation=False, padding=False) #TAIL
    else:
        '''tokens = []
        for i in range(0, len(train_texts), 1): #, had to split it by 5k otherwise failed ???
            tokens.append(tokenizer(train_texts[i: i+1], truncation=True, padding=True, max_length=int((SEG_SIZE-2)*NUM_SEGMENTS_FOR_DOCS+2)))
        train_encodings  = []
        for t1 in tokens:
            for t2 in t1:
                train_encodings.append(t2)'''
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=int((SEG_SIZE-2)*NUM_SEGMENTS_FOR_DOCS+2))
        for e in train_encodings["input_ids"]: assert len(e) == len(train_encodings["input_ids"][0])  #
        valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=int((SEG_SIZE-2)*NUM_SEGMENTS_FOR_DOCS+2))
        for e in valid_encodings["input_ids"]: assert len(e) == len(valid_encodings["input_ids"][0])  #
    train_encodings["input_ids"], train_encodings["token_type_ids"], train_encodings[
        "attention_mask"], train_labels = Segmentize(train_encodings["input_ids"], train_encodings["token_type_ids"],
                                                         train_encodings["attention_mask"], train_labels)
    valid_encodings["input_ids"], valid_encodings["token_type_ids"], valid_encodings[
        "attention_mask"], valid_labels = Segmentize(valid_encodings["input_ids"], valid_encodings["token_type_ids"],
                                                         valid_encodings["attention_mask"], valid_labels, val=True)

'''f = open(fast_path+"train-" + args.condition + "-tail2-pos-under-fast-ids.txt", "w") #B # DIS JULY 2022
for segs, lab in zip(train_encodings["input_ids"], train_labels):
    f.write(str(lab) + '\t' + ' '.join([str(s) for s in segs])+'\n')
f.close()
f = open(fast_path+"test-" + args.condition + "-tail2-pos-under-fast-ids.txt", "w") #B
for segs, lab in zip(valid_encodings["input_ids"], valid_labels):
    f.write(str(lab) + '\t' + ' '.join([str(s) for s in segs])+'\n')
f.close()
f=f
if args.data_only:
    sys.exit(0)'''

#ISSUE: length of input_ids has to match masks etc. otherwise they will not combine into batches, even while I am not using them in the calls to Bert
    #just fill in the masks once ids segmented?
#so the only solution is to segment the texts first, and then call tokenizer

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {k: torch.tensor(v[idx]).to(device) for k, v in self.encodings.items()} #
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        # item["labels"] = torch.tensor([self.labels[idx]], dtype = torch.long).to(device) #
        item["labels"] = torch.tensor([self.labels[idx]], dtype=torch.long)  #
        # item["labels"] = torch.tensor([self.labels[idx]])

        return item

    def __len__(self):
        return len(self.labels)


# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels) #label 'valid' inherited from tutorial for test set
#TestLen = int (len(valid_labels)) #SEG *2/3 in cats #VALID moved up

class BertForSequenceClassificationCustom(BertForSequenceClassification):
#BertForSequenceClassification.from_pretrained(model_name, num_labels=len(Labels))
    '''def __init__(self, config, **kwargs): #unlike A P trying pure inheritance: not containing Bert and not changing anything
        super(BertForSequenceClassification, self).__init__(self, config, **kwargs) #errs
        #super(BertForSequenceClassification, self).__init__(config) #from A P, errs. not clear why he has BertForSequenceClassification as superclass for BertForSequenceClassification???'''

    '''def from_pretrained(self, model_name, num_labels): #could not get this to work, 'cls' in org code?? this somehoe needs to be marked as a constructor, but syntax for that not clear
        self.classifier = torch.nn.Linear(768, num_labels)
        #INCOMPLETE: dropout
        return BertForSequenceClassification.from_pretrained(model_name, num_labels=len(Labels))'''


    '''def _init_weights(self, module): #INIT #LAPTOP dis don't need over-ride if not playing with init ranges
        """ Initialize the weights """  #CARE: this is copied form web, could be earlier version ??
        from torch import nn #INIT
        if isinstance(module, (nn.Linear, nn.Embedding)): #why pair () ??
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=args.initializer_range) #this one is called for this class
            #module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) #this one is called for this class
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()'''


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

     SequenceLength = SEG_SIZE #SEG
     #assert len(input_ids) == 1 #TEMP batch size
     batch_out = []

     #unfrozen = random.randint(0,len(input_ids)) #
     #assert len(input_ids) == args.per_device_train_batch_size #fails since some batches are not complete?
     assert len(input_ids) <= args.per_device_train_batch_size #
     unfrozen = np.random.randint(low=0, high=len(input_ids), # moved here
                                  size=(args.unfrozen_segs,))  # : batches are frozen, not segment
     # unfrozen = np.random.randint(low=0, high=NUM_SEGMENTS_FOR_DOCS, size=(args.unfrozen_segs,))  # #INCOMPLETE: need to sample without repeats##
     assert len(input_ids) > 0 # JULY 2022
     for b in range(len(input_ids)): #batches loop BATCH
     #for b in range(args.per_device_train_batch_size): #batches loop BATCH
        num_non_empty_segments = 0
        for i in range(NUM_SEGMENTS_FOR_DOCS):  #LAPTOP
            assert input_ids[b][SequenceLength*i] == 101
            if input_ids[b][SequenceLength*i+3] == 0 and i == 0:
                i = i
            if False: # JULY 2022, was it a bug ??? seems entire batch skipped if has a single padded doc
            #if input_ids[b][SequenceLength*i+3] == 0: #padding starts right after
                assert attention_mask[b][SequenceLength*i+3] == 0
                break
            else:
                num_non_empty_segments += 1
                if attention_mask[b][SequenceLength * i + 3] != 1:
                     i = i
                #assert attention_mask[b][SequenceLength * i + 3] == 1 # JULY 2020 dis, was it a bug above ??


        segs = []
        masks = []
        assert num_non_empty_segments > 0 # JULY 2022
        for i in range(num_non_empty_segments):  #LAPTOP
        #for i in range(NUM_SEGMENTS_FOR_DOCS):  #LAPTOP
            segs.append(input_ids[b][SequenceLength*i:SequenceLength*i + SequenceLength]) #BATCH
            #segs.append(input_ids[0][SequenceLength*i:SequenceLength*i + SequenceLength])
            masks.append(attention_mask[b][SequenceLength*i:SequenceLength*i + SequenceLength]) #BATCH #MASK
            #assert not (0 in attention_mask[b][SequenceLength*i:SequenceLength*i + SequenceLength]) #MASK dis, since would fail in incomplete segments
        assert len(segs) # JULY 2022
        assert len(masks) # JULY 2022
        seqsb = torch.stack(segs)
        seqsm = torch.stack(masks)
        if True: #
        #if b in unfrozen: #
        #if b == unfrozen:
                for e in self.bert.parameters(): e.requires_grad = True
        else:
                for e in self.bert.parameters(): e.requires_grad = False
        if args.frieze_embeds:
            modules = [self.bert.embeddings, *self.bert.encoder.layer[:args.layer_frozen]] #works, what * means ???
        else:
            modules = [*self.bert.encoder.layer[:args.layer_frozen]]  # works, what * means ???
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        out = self.bert(seqsb, output_hidden_states=True, attention_mask = seqsm).hidden_states[args.layer] #MASK
        #BATCH SLOW: batches can be also added to the same call to bert(), or to two calls: (un-)frozen
        if args.layer == 0:
            out_cls = out.sum(1) #mean pool of the layer #
        else:
            out_cls = out[:,0,:] #LAPTOP CLS tokens
        out1 = out_cls.sum(0) #LAPTOP pooling the segments
        out1 = out1 / num_non_empty_segments ##MASK
        #out1 = out1 / SEG_SIZE #LAPTOP should be NUM_SEGMENTS_FOR_DOCS anyway?
        batch_out.append(out1) #BATCH
     out1 = torch.stack(batch_out)
     classifier_out = self.classifier(self.cdropout(out1)) #BATCH, no [] around out1
     #classifier_out = self.classifier(self.cdropout(torch.stack([out1]))) #
     logits = classifier_out #they seem like logits, not probs - but errs later
     outputs = (logits,)  #SEG INCOMPLETE dis; since org below errs due to change of 'outputs' dimensions, works so far

     if labels is not None:
            #assert False #this code runs
            #loss_fct = BCELoss() #errs, likely needs changing sizes to 1D
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

     return outputs  # (loss), logits, (hidden_states), (attentions)
        #'''


        #return  BertForSequenceClassification.forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
            #works and produces identical result to using Bert Class directly

# load the model and pass to CUDA
model = BertForSequenceClassificationCustom.from_pretrained(model_name, num_labels=10 if not args.binary else 2)  # JULY 2022
#model = BertForSequenceClassificationCustom.from_pretrained(model_name, num_labels=2)  # SEG
model.cdropout = torch.nn.Dropout(.1) #

#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device) #
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to(device)
# model.to(device) #

model_label = '-l' if args.bert_model == "bert-large-uncased" else '-s'
ResulFileName = os.path.basename(__file__)  + ".lr" + str(args.learning_rate) + ".b" + str(args.gradient_accumulation_steps)  + '.' + str(NUM_SEGMENTS_FOR_DOCS) + 'x'+  str(SEG_SIZE) + 's' + '.' + str(args.layer) + ".data.txt"
#ResulFileName = os.path.basename(__file__) + '.'+ args.condition + ".lr" + str(args.learning_rate) + ".b" + str(args.gradient_accumulation_steps)  + '.' + str(NUM_SEGMENTS_FOR_DOCS) + 'x'+  str(SEG_SIZE) + 's' + '.' + str(args.layer) + ".data.txt"
already_reported = False
eval_count = 0
test_roc_best = 0
not_increased = 0
vf1_max = 0.
f1_max = 0.


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
  #print(confusion_matrix(labels[:TestLen], preds[:TestLen],labels=[i for i in range(10)]))  #
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

training_args = TrainingArguments(
    save_total_limit = 1,
    resume_from_checkpoint=args.resume_from_checkpoint,  #
    #resume_from_checkpoint=args.resume_from_checkpoint,  #SEG no effect here
    # resume_from_checkpoint = True,
    # resume_from_checkpoint = 'results-over/checkpoint-16000', #worked - No! it just does not affect anything
    learning_rate=args.learning_rate,  # 
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 
    per_device_train_batch_size= args.per_device_train_batch_size, #  1,  #GEN #BATCH
    output_dir=args.output_dir,  # VARIOUS
    # output_dir='./results-dead_365d-val-sep-under',          # output directory
    num_train_epochs=args.num_train_epochs,  # 
    # num_train_epochs=6,              #
    # per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=args.per_device_train_batch_size,  #SEG
    # per_device_eval_batch_size=20,   # batch size for evaluation
    #lr_scheduler_type='constant', #FLAT, verified that indeed stays so, so 'warmup_steps' and 'weight_decay' ignored 
    #warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler  - DR: this does not affect "multiples" error
    warmup_steps=args.warmup_steps,                # since
    #warmup_steps=500 if not args.resume_from_checkpoint else 0, # norm 500
    # number of warmup steps for learning rate scheduler  - DR: this does not affect "multiples" error
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    # logging_steps=500,               #leads to some portion runs not saved as a checkpoint
    eval_steps = args.logging_steps, # JULY 2022
    logging_steps=args.logging_steps,  # was used overnight but lead to 75% of time testing
    # logging_steps=1,               #since x8 larger #
    evaluation_strategy="steps",  # evaluate each `logging_steps`
    ignore_data_skip=True,  #  , was it to be able to re-set learn rate?
    save_steps=args.logging_steps if not args.eval_only else 10000000 ,  # JULY 2022
    #save_steps=100,  # A added only ~800 steps, and not yet re-started
)


trainer = Trainer(
    # place_model_on_device = "cpu", #DR: does not recog
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=valid_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    # resume_from_checkpoint=True, #keyword errs
    # resume_from_checkpoint='results-over\checkpoint-160000',  #errs
    # ignore_data_skip = True, #errs
) #LAPTOP INCOMPLETE: Noticed that shuffling is on by default, so can possibly command Trainer not to do it.
if args.flat_lr:
    training_args.lr_scheduler_type = 'constant' # 09/02/23

'''if args.eval_only:
    print(trainer.evaluate()) #works, but it does not load a model
    sys.exit(0)'''
# train the model
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=True)  #  norm empty
    #trainer.train(resume_from_checkpoint="checkpoint-15")  #  norm empty
    #trainer.train(resume_from_checkpoint=True)  #  norm empty
else:
    trainer.train()


if args.auto_val:
    print("\nmax val:", vf1_max, "max test:", f1_max,"\n")
    open(f"log{args.log_label}.txt",'a').write(f"\n{args},{f1_max}") 
    #open('log.txt','a').write(f"\n{args},{f1_max}") #
