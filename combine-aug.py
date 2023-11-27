# Copyright (C) 2022-2023  Dmitri Roussinov
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
This code supports additional dataset manipulations 
Posted on https://github.com/dminus1/genre
For the the following paper:
BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff, EMNLP Findings, 2023
'''
import random
seed_id = 0
random.seed(seed_id)
import collections
import re
import numpy
import os.path #
import sys #



import argparse
parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--out', type=str, default="shuffled-tmp.txt", help='')
parser.add_argument('--org', type=str, default="", help='')
parser.add_argument('--post_label', type=str, default="", help='') #

parser.add_argument('--label', type=str, default="", help='')
parser.add_argument('--merge', type=str, default="", help='')
parser.add_argument('--overlap_ok', action='store_true', help='')


parser.add_argument('--topic', type=int, default=1, help='genre index')
parser.add_argument('--cap', type=int, default=10000000, help='genre index') # 15/08 to avoid future accidental reductions
#parser.add_argument('--cap', type=int, default=100000, help='genre index')
parser.add_argument('--reduce', type=int, default=1, help='genre index')
parser.add_argument('--reduce_by', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by1', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by2', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by3', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by4', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by5', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by6', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by7', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by8', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by9', type=int, default=0, help='genre index')
parser.add_argument('--reduce_by10', type=int, default=0, help='genre index')

#parser.add_argument('--extra', type=str, default="", help='') "errs on -4e"
parser.add_argument('--preprocess', action='store_true', help='')
parser.add_argument('--binary', action='store_true', help='')
parser.add_argument('--sample', action='store_true', help='')
parser.add_argument('--balance', action='store_true', help='')
parser.add_argument('--balance_full', action='store_true', help='')
parser.add_argument('--shuffle', type=int, default=0, help='genre index') #, was false by default prior:
#parser.add_argument('--shuffle', action='store_true', help='')
parser.add_argument('--strip_id', action='store_true', help='')
parser.add_argument('--pattern',  type=str, default="", help='')
#parser.add_argument('--pre', type=str, default="-p", help='') #those with - don't work and quotes don't help ??
parser.add_argument('--nopunct', action='store_true', help='')
parser.add_argument('--no_org', action='store_true', help='')
parser.add_argument('--rep', type=int, default=1, help='genre index')
parser.add_argument('--filter', type=str, default="", help='')
parser.add_argument('--binarize', type=str, default="", help='')
parser.add_argument('--inp', type=str, default="", help='')
parser.add_argument('--exclude_ids', type=str, default="", help='')
parser.add_argument('--clean', action='store_true', help='')
parser.add_argument('--cut', action='store_true', help='')
parser.add_argument('--genre', action='store_true', help='')
parser.add_argument('--keywords', action='store_true', help='')
parser.add_argument('--train_tops', action='store_true', help='')
parser.add_argument('--strip', action='store_true', help='')
parser.add_argument('--strip_punct', action='store_true', help='')
parser.add_argument('--merge_binary', action='store_true', help='')
parser.add_argument('--strip_white', action='store_true', help='')
#parser.add_argument('--compensate', action='store_true', help='')
parser.add_argument('--compensate', type=int, default=0, help='genre index')
parser.add_argument('--gens', type=int, default=1, help='genre index')
parser.add_argument('--exclude_products', action='store_true', help='')
parser.add_argument('--merge_tests', action='store_true', help='')
parser.add_argument('--reduce_to', type=int, default=0, help='genre index')
parser.add_argument('--strip_empty', action='store_true', help='')
parser.add_argument('--sample_to_topics', action='store_true', help='')
parser.add_argument('--binarize_2', action='store_true', help='')
parser.add_argument('--no_cleaning', action='store_true', help='')
parser.add_argument('--stats', type=str, default="", help='')
parser.add_argument('--test_set', type=str, default="", help='')
parser.add_argument('--org_rep', type=int, default=1, help='genre index')
parser.add_argument('--shuf_gen_labels', action='store_true', help='')
parser.add_argument('--unique_ids', type=str, default="", help='')


args = parser.parse_args()

if args.unique_ids != "":
    IDs = collections.OrderedDict()
    for i in range(25):


        for line in open(f"classifier/data/test-single-topic-{i+1}-C1C2-1k-100-nop.tsv", encoding='utf-8').readlines()[1:]:
        #for line in open(f"classifier/data/train-topic{i+1}-reduced-topic-C1C2-10_b-no-punct.tsv", encoding='utf-8').readlines()[1:]:
        #for line in open(f"classifier/data/val-top-{i+1}-C1C2-100.tsv", encoding='utf-8').readlines()[1:]:
        #for line in open(f"classifier/data/train-bottom-{i+1}-{args.unique_ids}.tsv", encoding='utf-8').readlines()[1:]:
            id  = int(line.split('\t')[0])
            if id not in IDs:
                IDs[id] = len(IDs)
    print (len(IDs))
    sys.exit(0)

if args.exclude_ids != "":
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    ExcludedIDs = collections.OrderedDict()
    excluded = 0
    for line in open(args.exclude_ids,encoding='utf-8').readlines()[1:]:
        ExcludedIDs[int(line.split('\t')[0])] = len(ExcludedIDs)
    for line in open(args.inp,encoding='utf-8').readlines()[1:]:
        if int(line.split('\t')[0]) not in ExcludedIDs:
                    f.write(line)
        else:
            excluded+=1
    print("excluded: ", excluded)
    assert excluded == 100 #for 100-sample size 
    sys.exit(0)

if args.test_set != "" and args.reduce_to == 0: #check test-set not overlapping with train given in --inp
    OtherSet = collections.OrderedDict()
    for line in open(args.test_set, encoding='utf-8').readlines()[1:]:
        OtherSet[int(line.split('\t')[0])] = len(OtherSet)
    overlap_count = 0
    for line in open(args.inp, encoding='utf-8').readlines()[1:]:
        if int(line.split('\t')[0])  in OtherSet:
            overlap_count+= 1
            OtherSet = OtherSet

        #assert int(line.split('\t')[0]) not in OtherSet # dis
    print(f"overlap:{overlap_count}")
    sys.exit(0)

if args.binarize_2:
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    GenreCounts = [0 for _ in range(10)]  # 2022
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]
    for line in open(args.inp, encoding='utf-8').readlines()[1:]:
        genre_str = line.split('\t')[1]
        GenreID = Genres.index(genre_str)  # 2022
        if GenreID in [0,1]:
            f.write(line)
            GenreCounts[GenreID] += 1
    sys.exit(0)

if args.sample_to_topics:
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    TopicSizes = [0 for _ in range(1,26)]
    TopicSizes[10-1] = 17383
    TopicSizes[22-1] = 2558
    TopicSizes[24-1] = 24687


    #TopicSizes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #assert len(TopicSizes) == 25
    #TopicSizes = [802,4681,589,302,351,2016,188,2972,110,17383,349,1010,86,1111,2510,3262,2801,1840,548,2273,1907,2558,323,24687,80]
    MaxTopicSize = 24687
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]

    for i in range(1, 26):
        name = f"classifier/data/test-single-topic-{i}-random-test-ids-nopunct-30.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-1000-r-bytopics-nopunct-30.tsv" #
        count = 0
        for line in open(name, encoding='utf-8').readlines()[1:]:
            if args.binary: #
                genre_str = line.split('\t')[1]
                GenreID = Genres.index(genre_str)  # 2022
                if GenreID not in [0, 1]:
                    continue
            rep = TopicSizes[i-1]/MaxTopicSize
            #assert rep < 100
            for _ in range(6):
                if random.randint(0, 1000) < rep * 1000:
                    f.write(line)
                    count += 1
        print(count,)
    sys.exit(0)

if args.reduce_to != 0:
    GenreCounts = [0 for _ in range(10)] #2022
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    ft = open(args.test_set, "w", encoding='utf-8')
    ft.write("\ttarget_text\tinput_text\tprefix\n")
    lines = open(args.inp, encoding='utf-8').readlines()[1:]
    assert  "aug" not in args.out or len(lines) >= (30)*6 - 30 #30
    random.shuffle(lines) # 20/10/22
    written = 0
    for line in lines:
    #for line in open(args.inp, encoding='utf-8').readlines()[1:]:
        genre_str = line.split('\t')[1]
        GenreID = Genres.index(genre_str)  # 2022
        if GenreCounts[GenreID] < args.reduce_to and written < args.cap:
        #if GenreCounts[GenreID] < args.reduce_to:
            f.write(line)
            written+=1
        else: # 20/10/22
            ft.write(line)
        GenreCounts[GenreID] += 1
    sys.exit(0)


if args.merge_tests:
    lines = []
    #for i in [16, 17, 20]:
    #for i in[1, 2, 6, 8, 12, 14]: #
    for i in [2,10,16,17,18,20,24, 15,22,21,12,14]:
    #for i in range(1, 26):
     #if i not in [10,24,16,17,20]:
        #name = f"classifier/data/test-single-topic-{i}-random-test-ids-nopunct.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-random-test-ids-nopunct-30.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-1000-r-bytopics-nopunct-30.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-500-r-bytopics.tsv" #
        name = f"classifier/data/test-single-topic-{i}-500-r-thresh-100.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-500-r-bytopics-100.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-500-r-bytopics-30.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-1000-r-nopunct-30.tsv" #
        #name = f"classifier/data/test-single-topic-{i}-1000-r-nopunct.tsv"
        if os.path.exists(name):
            lines   += open(name).readlines()[1:]
        else:
            print(f"missing {name}")
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in lines:
        f.write(line)
    sys.exit(0)


#CARE: copied from from-SS-genre-to-t5.py, may need updating to be consistent
def Clean(line): return line.replace("\"", ' \" ').replace(".", ' . ').replace("\/", ' \/ ').replace("!", ' ! ').replace("?", ' ? ').replace(":", ' : ').replace(";", ' ; ').replace("(", ' ( ').replace(")", ' ) ').replace("\n", ' ').replace("\t", ' ').replace("*", ' * ').replace(",", ' , ').replace("-", ' - ').lower() #, added '-'

def CleanText(text):
    tmp = []  # pre-processing
    for c in text.lower():  # same pre-processing as inside from-SS-genre-to-t5-bp.py
        # tmp.append(c if 'a' <= c <= 'z' or c in ['.', ',',';','-','?','!',"'",'"'] else ' ') #
        if 'a' <= c <= 'z' or c == ' ':
            tmp.append(c)
        # tmp.append(c if 'a' <= c <= 'z' else ' ') #did for experiments prior to 30/05/22
        if not args.nopunct:  # 
            if c in ['.', ',', ';', '-', '?', '!', "'", '"']: #23/06/22
            #if c in [' ', '.', ',', ';', '-', '?', '!', "'", '"']:
                #tmp.append(' ') #
                tmp.append(c)
                #tmp.append(' ') #
        else:
            if c in ['.', ',', ';', '-', '?', '!', "'", '"']:
            #if c in [' ', '.', ',', ';', '-', '?', '!', "'", '"']:
                tmp.append(' ')
    text = ''.join(tmp)

    rest_of_line = text  # extra filtering added on 21/06, copied from "from-SS-genre-to-t5.py
    rest_of_line = rest_of_line[5:] if rest_of_line[:5] == '  q h' else rest_of_line  # for stack
    while "h history h" in rest_of_line:  # for wiki #2022
        rest_of_line = rest_of_line.replace("h history h", " ")
    rest_of_line_list = rest_of_line.split()
    while 'h' in rest_of_line_list:
        rest_of_line_list.remove('h')
    while 'p' in rest_of_line_list:
        rest_of_line_list.remove('p')
    while 'b' in rest_of_line_list:
        rest_of_line_list.remove('b')
    while 'q' in rest_of_line_list: #added on 25/02
        rest_of_line_list.remove('q')
    rest_of_line = ' '.join(rest_of_line_list)
    # added later after .more model was already trained
    while "url afe" in rest_of_line:  #
        rest_of_line = rest_of_line.replace("url afe", " ")
    while "url \" afe . \"" in rest_of_line:  #added later for version splitting punctuation from the words
        rest_of_line = rest_of_line.replace("url \" afe . \"", " ")


    rest_of_line = rest_of_line[5:] if rest_of_line[:5] == '  q h' else rest_of_line  # for stack
    # extra filtering added on 20/06, copied from "from-SS-genre-to-t5.py
    # "

    # added on 21/06/22 to be consistent with from-SS-genre-to-t5.py:
    rest_of_line = re.sub('doc url\".*', '', rest_of_line)
    rest_of_line = re.sub('doc url \".*', '', rest_of_line) #added later
    # rest_of_line = re.sub('doc url\".*\"', '', rest_of_line)
    rest_of_line = re.sub('url\".*', '', rest_of_line)
    rest_of_line = re.sub('url \".*', '', rest_of_line) #added later
    if "like us on facebook" in rest_of_line:
        rest_of_line = rest_of_line
    rest_of_line = re.sub('like us on facebook.*', ' ', rest_of_line)
    rest_of_line = re.sub('m-cm.* ', ' ', rest_of_line)
    rest_of_line = re.sub('m-bm.* ', ' ', rest_of_line)
    rest_of_line = re.sub("bamp\;q", ' ', rest_of_line)
    rest_of_line = re.sub("bamp \; q", ' ', rest_of_line) #added later
    rest_of_line = re.sub("nbsp ; ", ' ', rest_of_line)
    text = rest_of_line
    return text

if args.strip_id : #merges t
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    lines = []
    count = 0
    for line in open(args.inp, encoding='utf-8').readlines()[1:]: #input is the aug gen
        part = line.split('\t')
        part[0] = str(count)
        #part[0] = '0'
        part[1] = 'dummy'
        f.write('\t'.join(part))
        count += 1
    sys.exit(0)

if args.merge_binary : #merges t
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    lines = []
    for line in open(args.inp, encoding='utf-8').readlines()[1:]: #input is the aug gen
        part = line.split('\t')
        assert part[1] == 'dummy'
        part[1] = '1'
        part[2] = Clean(CleanText(part[2]))
        assert part[3] == 'summarize\n'
        part[3] = 'genre\n'
        lines.append('\t'.join(part))
    lines_org = [line for line in open(args.merge, encoding='utf-8').readlines()[1:] if line.split('\t')[1] == '0']
    random.shuffle(lines_org)
    lines += lines_org[:len(lines)]
    random.shuffle(lines)
    for line in lines:
        f.write(line)
    sys.exit(0)


if args.binarize != '':
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    pos = [] ; neg = []
    for line in open(args.inp, encoding='utf-8').readlines()[1:args.cap + 1]:
        part = line.split('\t')
        if part[1] == args.binarize:
            part[1] = '1'
            pos.append('\t'.join(part))
        else:
            part[1] = '0'
            neg.append('\t'.join(part))
    random.shuffle(pos)
    random.shuffle(neg)
    mins = min(len(pos), len(neg))
    both = pos[:mins] + neg[:mins]
    random.shuffle(both)
    for line in both:
        f.write(line)
    sys.exit(0)

def id(line):
    return int(line.split('\t')[0])


if args.merge != '': #merge given data file with input and shuffle
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    lines = []
    OtherSet = collections.OrderedDict()
    for _ in range (args.org_rep): 
        for line in open(args.inp, encoding='utf-8').readlines()[1:]:
            if len(line) > 10: #
                lines.append(line.replace('\n', '') + '\n') #
                #lines.append(line)
                if id(line) not in OtherSet:
                    OtherSet[id(line)] = len(OtherSet)
    random.shuffle(lines)
    #lines += lines[:len(open(args.merge, encoding='utf-8').readlines())]# dis 06/07 realized that this does not balance anyway, ignoring for now. 200/10000 = 2%, so 2 etra hypers out of 100, can not explain the fix, also will lead to reduction of other labels - but it may promote specificially for the excluded topic?
    for line in open(args.merge, encoding='utf-8').readlines()[1:]:
        if len(line) > 10: #
            lines.append(line.replace('\n', '') + '\n') # dis
        if not args.overlap_ok:
            if not (id(line) not in OtherSet):
                print(line)
            assert id(line) not in OtherSet
        #lines.append(Clean(CleanText(line.replace('\n', ''))))
    random.shuffle(lines)
    for line in lines:
        f.write(line)
    sys.exit(0)



if args.strip_white:
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in open(args.inp, encoding='utf-8').readlines()[1:args.cap + 1]:
        part = line.split('\t')

        tmp = []  # pre-processing
        text = part[2]
        text = ' '.join(text.split())
        #text = re.sub('[ ].+', ' ', text)
        part[2] = text
        f.write('\t'.join(part))
    sys.exit(0)


if args.strip_empty:
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in open(args.inp, encoding='utf-8').readlines()[1:args.cap + 1]:
        part = line.split('\t')

        tmp = []  # pre-processing
        text = part[2]
        while '  ' in text:
            text = text.replace('  ', ' ')
        part[2] = text
        f.write('\t'.join(part))
    sys.exit(0)

if args.strip_punct:
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    lines = open(args.inp, encoding='utf-8').readlines()
    for line in lines[1:args.cap + 1]:
        part = line.split('\t')

        tmp = []  # pre-processing
        text = part[2]
        for c in text.lower():  # same pre-processing as inside from-SS-genre-to-t5-bp.py
            # tmp.append(c if 'a' <= c <= 'z' or c in ['.', ',',';','-','?','!',"'",'"'] else ' ') #
            if 'a' <= c <= 'z' or c == ' ':
                tmp.append(c)
            if c in ['.', ',', ';', '-', '?', '!', "'", '"']:
                    tmp.append(' ')
        text = ''.join(tmp)
        part[2] = text
        f.write('\t'.join(part))
    print(f"lines:{len(lines)}")
    sys.exit(0)

if args.strip: #removes keywords from given LM train file
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in open(args.inp, encoding='utf-8').readlines()[1:args.cap + 1]:
        part = line.split('\t')
        f.write(part[0] + '\t' + part[1]  + '\tdummy\t' + part[3])
    sys.exit(0)
#extract keywords from old data using new set of keywords
if args.keywords:
    # new_keywords = []
    new_keywords =  collections.OrderedDict()
    for line in open("aug-keywords-8-4000-for-filtering-100.tsv", encoding='utf-8').readlines()[1:]:
        new_keywords[id(line)] = len(new_keywords)
    genres = ['hyper', 'giga','reviews', 'products', 'stories', 'wiki', 'stack']
    f = open("out.txt", "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    lines = []
    for g in genres:
        for line in open("train-" + g + "-topLDA.tsv", encoding='utf-8').readlines()[1:]:
            if id(line) in new_keywords:
                lines.append(line)
    random.shuffle(lines)
    for line in lines[:args.cap]:
                f.write(line)
    sys.exit(0)

#extract first lines from each genre - not needed since same for all topics - but the size ? also train files don't have excluded topic docs - so see code below


if args.train_tops:
    genres = ['hyper', 'stories', 'products', 'wiki',  'reviews',  'stack', 'giga'] #changed to standard order on 26/06 after hyper already gen-ed
    #genres = ['hyper', 'giga', 'reviews', 'products', 'stories', 'wiki', 'stack']
    f = open("out.txt", "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for g in genres:
            for line in open("train-" + g + "-12-2000.txt", encoding='utf-8').readlines()[1:args.cap+1]:
                f.write(line)
    sys.exit(0)

#extract genre classifier datasets from LM datafiles
if args.genre:
    lines = []
    genres = ['hyper', 'giga','reviews', 'products', 'stories', 'wiki', 'stack']
    labels = ["__id__A01-discuss-hyper-sample.ol","__id__A8-giga-en.clean.ol", "__id__A17-review-sample.ol", "__id__A12-all", "__id__A11-icwsm09stories.real.ol", "__id__A16-wiki.ol",  "__id__A7-stackexchange-sample.ol"]

    for g, label in zip(genres, labels):
        for line in open("train-"+g+"-topLDA.tsv", encoding='utf-8').readlines()[1:args.cap+1]:
            part = line.split('\t')
            lines.append(part[0]+'\t'+label + '\t' + Clean(CleanText(part[1])) + '\t' + "genre\n")
    random.shuffle(lines)
    f = open("out.txt", "w", encoding='utf-8')
    fte = open("outte.txt", "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in lines[:-1000]:
        f.write(line)
    for line in lines[-1000:]:
        fte.write(line)
    sys.exit(0)

#balance given dataset with genre labels
if args.pattern != '':
    GenreCounts = [0 for _ in range(10)] #2022
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]
    for line in open("classifier/data/test-2000.tsv", encoding='utf-8').readlines()[1:]:
    #for line in open("classifier/data/genre-classifier-test-4000-4k-clean.txt", encoding='utf-8').readlines()[1:]:
        if args.pattern in line:
            genre_str = line.split('\t')[1]
            if genre_str == '__id__A12-all':
                genre_str = '__id__A12-all.ol'
            GenreID = Genres.index(genre_str)  # 2022
            GenreCounts[GenreID] += 1
    print (GenreCounts)
    sys.exit(0)

if args.balance_full:
    GenreCounts = [0 for _ in range(10)] #2022
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]
    for line in open(args.inp, encoding='utf-8').readlines()[1:]:
    #for line in open("classifier/data/train-1000-more-cleaning4-base.tsv", encoding='utf-8').readlines()[1:]:
        genre_str = line.split('\t')[1]
        GenreID = Genres.index(genre_str)  # 2022
        GenreCounts[GenreID] += 1
    maxSoFar = 0
    for GenreID in range(10):
        if  GenreCounts[GenreID] > maxSoFar:
            maxSoFar = GenreCounts[GenreID]
    assert maxSoFar > 0
    GenreCountsBalanced = [0 for _ in range(10)] #2022
    lines = []
    olines = open(args.inp, encoding='utf-8').readlines()[1:]
    random.shuffle(olines)
    for line in olines:
        genre_str = line.split('\t')[1]
        GenreID = Genres.index(genre_str)  # 2022
        rep  =  float(maxSoFar) / GenreCounts[GenreID]
        assert  rep < 1000 #C2
        #assert  rep < 100
        for _ in range (100):
            if random.randint(0, 1000) < rep/100. * 1000:
                GenreCountsBalanced[GenreID] += 1
                lines.append(line)
    print(GenreCountsBalanced)
    random.shuffle(lines)
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in lines:
            f.write(line)
    sys.exit(0)

if args.balance:
    GenreCounts = [0 for _ in range(10)] #2022
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]
    lines = []
    #for line in open("classifier/data/test-1000-more-cleaning4-base.tsv", encoding='utf-8').readlines()[1:]:
    olines = open(args.inp, encoding='utf-8').readlines()[1:] #
    #olines = open("classifier/data/test-single-topic-excluded-8-4000-for-filtering.tsv", encoding='utf-8').readlines()[1:]
    if args.shuffle:
        random.shuffle(olines)
    for line in olines:
    #for line in open("classifier/data/train-1000-more-cleaning4-base.tsv", encoding='utf-8').readlines()[1:]:
        genre_str = line.split('\t')[1]
        GenreID = Genres.index(genre_str)  # 2022
        GenreCounts[GenreID] += 1
        if GenreCounts[GenreID] <= args.cap:
            parts = line.split('\t')
            if not args.no_cleaning:
                parts[2] = Clean(CleanText(parts[2])) #added on 24/06
            line = '\t'.join(parts)
            lines.append(line)
    random.shuffle(lines)
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in lines:
            f.write(line)
    sys.exit(0)

if args.stats != '' or args.reduce_by1  != 0 or args.reduce_by2 != 0 or args.reduce_by3 != 0 or args.reduce_by4 != 0 or args.reduce_by6 != 0 or args.reduce_by7 != 0 :
#if args.reduce_by1  != 0 or args.reduce_by2 != 0:
 if args.stats != '':
    Genres = [g.replace('\n', '') for g in open("all-genre.txt").readlines()]
    GenreCountsTarget = [0 for _ in range(10)] #2022
    GenreCountsEmpty = [0 for _ in range(10)] #2022
    for line in open(args.stats, encoding='utf-8').readlines()[1:]:
         genre_str = line.split('\t')[1]
         GenreID = Genres.index(genre_str)  # 2022
         GenreCountsTarget[GenreID] += 1
         if len(line.split('\t')[2]) < 100:
             GenreCountsEmpty[GenreID] += 1
    if args.inp!= '':
        f = open(args.out, "w", encoding='utf-8')
        lines = open(args.inp, encoding='utf-8').readlines()
        print(f"org: {len(lines)}")
        f.write(lines[0])
        GenreCounts = [0 for _ in range(10)] #2022
        for line in lines[1:]:
            genre_str = line.split('\t')[1]
            GenreID = Genres.index(genre_str)  # 2022
            if GenreCounts[GenreID] < GenreCountsTarget[GenreID]:
                f.write(line)
                GenreCounts[GenreID] += 1
    print(GenreCountsTarget, GenreCountsEmpty)
 else:
    f = open(args.out, "w", encoding='utf-8')
    GenreCounts = [0 for _ in range(10)] #2022
    lines = open(args.inp, encoding='utf-8').readlines()
    print(f"org: {len(lines)}")
    f.write(lines[0])
    for line in lines[1:len(lines)-args.reduce_by]:
        genre_str = line.split('\t')[1]
        GenreID = Genres.index(genre_str)  # 2022
        GenreCounts[GenreID] += 1
        if GenreID == 0: #skipping first specified #docs, so assumed inputs already shuffled
            if GenreCounts[0] > args.reduce_by1:
                f.write(line)

        if GenreID == 1:
            if GenreCounts[1] > args.reduce_by2:
                f.write(line)

        if GenreID == 2:
            if GenreCounts[1] > args.reduce_by3:
                f.write(line)

        if GenreID == 3:
            if GenreCounts[1] > args.reduce_by4:
                f.write(line)

        if GenreID == 4:
            if GenreCounts[1] > args.reduce_by5:
                f.write(line)

        if GenreID == 5:
            if GenreCounts[1] > args.reduce_by6:
                f.write(line)

        if GenreID == 6:
            if GenreCounts[1] > args.reduce_by7:
                f.write(line)

        if GenreID == 7:
            if GenreCounts[1] > args.reduce_by8:
                f.write(line)

        if GenreID == 8:
            if GenreCounts[1] > args.reduce_by9:
                f.write(line)

        if GenreID == 9:
            if GenreCounts[1] > args.reduce_by10:
                f.write(line)

    print(GenreCountsTarget, GenreCounts)
 sys.exit(0)

#cut given number of lines
if args.reduce_by != 0:
    f = open(args.out, "w", encoding='utf-8')
    lines = open(args.inp, encoding='utf-8').readlines()
    print(f"org: {len(lines)}")
    f.write(lines[0])
    for line in lines[1:len(lines)-args.reduce_by]:
        f.write(line)
    sys.exit(0)

if args.cut:
    f = open(args.out, "w", encoding='utf-8')
    lines = open(args.inp, encoding='utf-8').readlines()
    f.write(lines[0])
    for line in lines[1:args.cap+1]:
        f.write(line)
    sys.exit(0)

#filter out old dataset based on present in new dataset

if args.filter != '':
 import collections
 Keep = collections.OrderedDict()
 for line in open(args.filter, encoding='utf-8').readlines()[1:]:
     id = int(line.split('\t')[0])
     Keep[id] = len(Keep)

 f = open(args.out, "w", encoding='utf-8')
 lines = open(args.inp, encoding='utf-8').readlines()
 f.write(lines[0])
 count = 0
 for line in lines[1:]:
     id = int(line.split('\t')[0])
     if id  in Keep and count < args.cap:
         f.write(line)
         count += 1
     else:
         count = count
 f.close()
 sys.exit(0)

#balance genre train set
'''GenreCount = [0] * 10
lines = open("classifier\data\\train-single-topic-excluded-10-4000o-beg.tsv", encoding='utf-8').readlines()
f = open("tmp.txt", "w", encoding='utf-8')
f.write(lines[0])
Genres = open("all-genre.txt").readlines()
Genres = [g.replace('\n','') for g in Genres]
for line in lines[1:]:
    GenreID = Genres.index(line.split('\t')[1])  # 
    if GenreCount[GenreID] < 4385:
        GenreCount[GenreID] += 1
        f.write(line)
f.close()
sys.exit(0)'''

#remove punctuation from a given datafile
if args.preprocess:

    #lines = open("classifier\data\\genre-classifier-test-4000.tsv", encoding='utf-8').readlines()
    lines = open(args.inp, encoding='utf-8').readlines()
    #lines = open("classifier\data\\genre-classifier-train-4000-10k.tsv", encoding='utf-8').readlines()
    f = open(args.out, "w", encoding='utf-8')
    #f = open("tmp.txt", "w", encoding='utf-8')
    f.write(lines[0])
    for line in lines[1:]:
        parts = line.split('\t')
        parts[2] = CleanText(parts[2])
        parts[2] = Clean(parts[2])
        f.write('\t'.join(parts))
    f.close()
    sys.exit(0)


#org data
i = args.topic
lines = []
if not args.no_org or args.compensate > 0: #
    #"train-single-topic-excluded-12-1000-r.tsv"
    if args.org == '':
        args.org = f"classifier/data/train-topic{i}-excluded-topic-500-r-thresh_b-no-punct.tsv"
    org_lines = open(args.org).readlines()[1:] #for 10 and 12
    #org_lines = open(f"classifier/data/train-topic{i}-excluded-topic-500-r-thresh_b-no-punct.tsv").readlines()[1:] #for 10 and 12
    #org_lines = open("classifier/data/train-single-topic-excluded-"+str(i)+"-1000-r-nopunct.tsv").readlines()[1:] #for 10 and 12
    #org_lines = open("classifier/data/train-single-topic-excluded-"+str(i)+"-10keys-nopunct.tsv").readlines()[1:] #for 10 and 12
    #org_lines = open("classifier/data/train-single-topic-excluded-"+str(i)+"-2000-nopunct.tsv").readlines()[1:] #for 8
    #org_lines = open("classifier\data\\train-single-topic-excluded-"+str(i)+"-b.tsv").readlines()
    #org_lines = open("classifier\data\\train-single-topic-excluded-"+str(i)+".tsv").readlines()
    #header = org_lines[0]
    if not args.no_org: #
        for _ in range (args.org_rep):
            lines += org_lines[0:int(len(org_lines)/args.reduce)]
    #lines += org_lines[0:int(len(org_lines)/4)]
    random.shuffle(lines) # 10/07
    #lines+=lines[:args.compensate]

header = "\ttarget_text\tinput_text\tprefix\n"
#lines += org_lines[1:]

#pre = "-bp-base-4e"
#pre = args.pre #does not work, see above
def AddLines(glabel, label):
 out = []  #  renamed from  'tmp', how did it work before?
 for gen in range(args.gens):
    '''if gen == 0:
     name = "train-aug-"+glabel+"-"+str(i)+"-" + args.label + ".txt " #
    else:'''
    #name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label + "." + str(gen+1) +  f"{args.post_label}-no-val-base.txt "  #
    name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label + "." + str(gen+1) +  f"{args.post_label}.txt "  #norm
    #name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label + "." + str(gen+1) +  f"{args.post_label}.bottom.txt "  #
    #name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label + "." + str(gen+1) +  f"{args.post_label}-no-val.txt"  #some gens need that
    #name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label + "." + str(gen+1) + ".txt "  #
    '''if args.gens == 1:
        name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label +  ".txt "  #'''

    # "train-aug-giga-12-500-r-thresh-30x6.1"
    #name = "train-aug-" + glabel + "-" + str(i) + "-" + args.label + "-" + str(gen+1) + ".txt "  #
    #name = "train-aug-"+glabel+"-"+str(i)+"-7g-self-" + args.label + ".txt " #
    #name = "train-aug-"+glabel+"-"+str(i)+"-7g-base1e-self.txt " #
    if not os.path.exists(name) or "stories" in name:
    #if not os.path.exists(name):
        print("missing "+ name)
        if args.compensate > 1:
            for _ in range(args.compensate):
                while True:
                    line = org_lines[random.randint(0, len(org_lines)-1)]
                    if line.split('\t')[1] == label:
                        out.append(line)
                        break
    else:
        alines = open(name, encoding='utf-8').readlines()[1:] #
        #alines = open("train-aug-"+glabel+"-"+str(i)+"-r1500-base1e.txt ").readlines()[1:]
        #alines = open("train-aug-"+glabel+"-"+str(i)+"-7g-base-4e.txt").readlines()[1:]
        # random.shuffle(alines) #  dis on 24/06
        #for j in range(len(alines[:args.cap])): #re-using old gens #
        for line in alines[:args.cap]: #re-using old gens #
        #for line in open("train-aug-"+glabel+"-"+str(i)+"-7g-base-4e.txt").readlines()[1:]: #re-using old gens #

        #for line in open("train-aug-"+glabel+"-"+str(i)+".txt").readlines()[1:]: #re-using old gens #
        #for line in open("train-aug-"+glabel+"-"+str(i)+"-7g-small.txt").readlines()[1:]: # - norm  for fresh gens on 10/06
        #for line in open("train-aug-"+glabel+"-"+str(i)+pre+".txt").readlines()[1:]:
        #for line in open("train-aug-"+glabel+"-"+str(i)+"-p.txt").readlines()[1:]:
        #for line in open("train-aug-"+glabel+"-"+str(i)+"-4e.txt").readlines()[1:]:#
        #for line in open("train-aug-"+glabel+"-"+str(i)+".txt").readlines()[1:]:
            #line = alines[j]
            parts = line.split('\t')
            assert 0 <= int(parts[0]) <= 5000
            #assert len(parts[2]) > 200 some are shorter
            assert parts[1]  == 'dummy'
            assert parts[3]  == 'summarize\n'
            text = CleanText(parts[2])
            if args.clean:
                text = Clean(text)
            out.append(parts[0]+ '\t'+ label + '\t' + text + '\tgenre\n') #
            #out.append('0\t'+ label + '\t' + text + '\tgenre\n')
    out = out      #
 return out


elines  = []
for _ in range(args.rep):
#for _ in range(0 if args.binary else 0): #
#for _ in range(1 if args.binary else 4 ):
    elines += AddLines('hyper', '__id__A01-discuss-hyper-sample.ol')
    elines += AddLines('giga', '__id__A8-giga-en.clean.ol')
    if not args.binary:
        elines += AddLines('wiki', '__id__A16-wiki.ol')
        elines += AddLines('reviews', '__id__A17-review-sample.ol')
        elines += AddLines('stack', '__id__A7-stackexchange-sample.ol')
        if not args.exclude_products:
            elines += AddLines('products', '__id__A12-all.ol') # 07/12 # CARE if re-running prior datasets
        elines += AddLines('stories', '__id__A11-icwsm09stories.real.ol')

if args.shuf_gen_labels:  # 
        assert args.rep == 1
        labels = []
        for j in range(len(elines)):
            parts = elines[j].split('\t')
            labels.append(parts[1])
        random.shuffle(labels)
        for j in range(len(elines)):
            parts = elines[j].split('\t')
            parts[1] = labels[j]
            elines[j] =  '\t'.join(parts)
lines  += elines
for _ in range(args.shuffle): #
#if args.shuffle:
    random.shuffle(lines) # dis
#random.shuffle(lines) #dis on 23/06 since so far used only for testing, and easier to keep track
f = open(args.out, "w", encoding='utf-8')
f.write(header)
for line in lines:
    f.write(line)
f.close()


