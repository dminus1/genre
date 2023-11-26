# Copyright (C) 2019  Dmitri Roussinov
# This program is free software under GPL 3, see http://www.gnu.org/licenses/
'''
This code creates on-topic and off-topic datasets based on the corpora with several genres.
Posted on https://github.com/dminus1/genre
For the the following paper:
BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff, EMNLP Findings, 2023
'''

import random
seed_id = 8
random.seed(seed_id)
import collections
import re
import numpy
import sys

prefix  = "expand"
#prefix  = "summarize" #=
#prefix = "deglue" #
UseTopics = True

import argparse #RECENT
parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--gi', type=int, default=0, help='genre index')
parser.add_argument('--data', action='store_false', help='')
#parser.add_argument('--data', action='store_true', help='')
parser.add_argument('--topic', type=int, default=1, help='genre index')
parser.add_argument('--aug', action='store_true', help='')
parser.add_argument('--punct', action='store_true', help='')
#parser.add_argument('--cap', type=int, default=10000000, help='genre index') #20/08 RECENT C2 dis
#parser.add_argument('--cap', type=int, default=100, help='genre index') #10/06 RECENT
parser.add_argument('--cut', type=int, default=1000, help='genre index') #10/06 RECENT
parser.add_argument('--random_cut', action='store_true', help='')
parser.add_argument('--self', type=str, default='', help='')
parser.add_argument('--ids_exclude', type=str, default='', help='')
parser.add_argument('--label', type=str, default='', help='')
parser.add_argument('--overall', action='store_true', help='create datasets for genre train/test without any topic exclusion')
parser.add_argument('--regen', action='store_true', help='create datasets for genre train/test without any topic exclusion')
parser.add_argument('--balanced_keywords', action='store_true', help='create datasets for genre train/test without any topic exclusion')
parser.add_argument('--keywords_from_genre_test', action='store_true', help='create datasets for genre train/test without any topic exclusion')
parser.add_argument('--tops', type=int, default=40, help='genre index') #10/06 RECENT
parser.add_argument('--genre_cap', type=int, default=10000000, help='genre index')
parser.add_argument('--thresh', type=float, default=.1, help='genre index') #RECENT
parser.add_argument('--data_thresh', type=float, default=.04, help='genre index') #RECENT
parser.add_argument('--by_topics',  action='store_true', help='create datasets for genre train/test without any topic exclusion')
parser.add_argument('--topic_class',  action='store_true', help='create datasets for genre train/test without any topic exclusion')
parser.add_argument('--topic_stats', action='store_true', help='')
parser.add_argument('--overall_genre', action='store_true', help='')
parser.add_argument('--inp', type=str, default="", help='')
parser.add_argument('--out', type=str, default="out-tmp.txt", help='')
parser.add_argument('--keywords_from_test', action='store_true', help='')
parser.add_argument('--keywords_from_test_split', action='store_true', help='')
parser.add_argument('--inverse_exclusion', action='store_true', help='')
parser.add_argument('--no_shuf', action='store_true', help='')
#parser.add_argument('--best_topic_test', action='store_true', help='')
parser.add_argument('--num_top', type=int, default=0, help='genre index')
parser.add_argument('--num_bottom', type=int, default=0, help='genre index')
parser.add_argument('--extract_doc', type=int, default=-1, help='genre index')

parser.add_argument('--top_topic_reserve', type=int, default=300, help='genre index')


args = parser.parse_args()



#re-write data with summaries, add genre level so can be used in a single model
labels = []
genre_label = "hyper" #0
labels.append(genre_label)
genre_label = "stories" #1
labels.append(genre_label)
genre_label = "A12" #2
labels.append(genre_label)
genre_label =  "arxiv" #3
labels.append(genre_label)
genre_label = "wiki" #4
labels.append(genre_label)
genre_label = "reviews" #5
labels.append(genre_label)
genre_label = "brown" #6
labels.append(genre_label)
genre_label = "stack" #7
labels.append(genre_label)
genre_label = "giga" #8
labels.append(genre_label)
genre_label = "legal" #9
labels.append(genre_label)
assert len(labels) == 10

counter = collections.Counter()  # C moved here
#indexing code to filter words by their DF-s
id2word  = [] #Mapping back from word ID-s to actual words
word2id = collections.OrderedDict()  # Python's class that can map words to ID-s
DFs = collections.OrderedDict()  # Python's class that can map words to ID-s
counter = collections.Counter()  # C moved here
word_count_list = counter.most_common()

def Clean(line): return line.replace("\"", ' \" ').replace(".", ' . ').replace("\/", ' \/ ').replace("!", ' ! ').replace("?", ' ? ').replace(":", ' : ').replace(";", ' ; ').replace("(", ' ( ').replace(")", ' ) ').replace("\n", ' ').replace("\t", ' ').replace("*", ' * ').replace(",", ' , ').replace("-", ' - ').lower() #RECENT, added '-'

#def Clean(line): return line.replace("\"", ' \" ').replace(".", ' . ').replace("\/", ' \/ ').replace("!", ' ! ').replace("?", ' ? ').replace(":", ' : ').replace(";", ' ; ').replace("(", ' ( ').replace(")", ' ) ').replace("\n", ' ').replace("\t", ' ').replace("*", ' * ').replace(",", ' , ').lower()

def create_dictionary(sentences): #this is indexing! we need to convert all words to their ID-s
    #counter = collections.Counter() #Python's class that can count
    for sentence in sentences:
        for word in sentence:
            counter.update([word])
            if word == "thares":
                word = word
    #word2id = collections.OrderedDict() #Python's class that can map words to ID-s
    word2id["<unk>"] = 0    #We reserve this for "uknown words" that we may encounter in the future
    word2id["<s>"] = 1 #Marks beginning of the sentence
    word2id["</s>"] = 2 #Marks the end of the sentence
    id2word.append("<unk>")
    id2word.append("<s>")
    id2word.append("</s>")
    word_count_list = counter.most_common() #For every word, we create an entry in  'word2id'
    for (word, count) in word_count_list: #so it can map them to their ID-s
            if word not in word2id: #this check important for consistency of mapping
                word2id[word] = len(word2id)
                id2word.append(word)

    for word in word2id: #Verifying that our mapping between words and their id-s is consistent
        assert id2word[word2id[word]] == word

    '''counter.update('c')
    counter.update('c')
    counter.update('c')
    counter.update('a')
    counter.update('b')
    counter.update('b')
    word_count_list = counter.most_common()
    sorted_tuples = []
    line = 'a b c d e' '''
    for (word, count) in word_count_list:  # so it can map them to their ID-s
        assert word not in DFs
        DFs[word] = count
    print("voc size:", len(DFs))
    #print(DFs.items())

    #return word2id, DFs

#
#'''
#Stops = []
Stops = collections.OrderedDict()
for line in open("NLTK-stopwords.txt").readlines():
    Stops[line.replace('\n', '')] = 1
Stops['+'] = 1
Stops['-'] = 1
Stops['!'] = 1
Stops[','] = 1
Stops[';'] = 1
Stops['.'] = 1
Stops['\t'] = 1
Stops['\"'] = 1
Stops['\''] = 1
Stops['<h>'] = 1
Stops['</h>'] = 1
EveryThLineUsed =   1 #
#IndexGenreUsed = [0] #
IndexGenreUsed = [0,1,4,5,7,8] #norm
#IndexGenreUsed = [0] #
#IndexGenreUsed = [0,1] #
#IndexGenreUsed = [1, 8]
C2CorpusPaths = ["discussion/A1-hyperpartisan.all", "personal/A11-icwsm09stories.sample2.all", "", "",
                 "info/A16-wikisample1.all", "reviews/A17-reviews1.all", "", "howto/A7-stackexchange.all",
                 "news/A8-giga-en.clean.all", ""]
C2GenreLabels = ["A1", "A11", "", "", "A16", "A17", "", "A7", "A8", ""]

#Read topics for each doc
CorporaUsed = [1,2] #norm

if UseTopics: # ok to dis when no keywords are generated, C2: < 1'
 TopicScore = [] #2D array: topic scores for for all docs, need to read it from a file
 DocMainTopic = []
 EmptyTopics = 0
 for gi in IndexGenreUsed:
 #for genre in C2GenreLabels:
  for corpus in CorporaUsed:
    genre = C2GenreLabels[gi]
    if corpus == 2:
     all_lines = open(f"S:/genre-corpus2/{genre}.full.txt").readlines() #C2
    else:
     all_lines = open(f"topics{gi}.txt").readlines()  # C1+C2

    #all_lines = open("S:/genre-corpus2/A17.full.txt").readlines() #C2
    #all_lines = open("ukwac-genres.25-all.topics.txt").readlines() #
    #all_lines = open("genre1.10.topics.txt").readlines()
    #assert len(all_lines) == 105216 #C2 dis
    TopicCounts = [0 for _ in range (25)] #C2
    skipped_empty = 0
    for docid in range(0, min(len(all_lines), args.genre_cap), 1):
    #for docid in range(0, len(all_lines), 1):
    #for docid in range(0, len(all_lines), EveryThLineUsed):
    #for line in all_lines:
        line = all_lines[docid] #RECENT
        #if line == '': continue #RECENT dis
        line_parts = line.split(',')
        #assert len(line_parts) == 50 #RECENT #C2 dis
        #assert len(line_parts) == 20
        topics_in_doc = 25 #RECENT
        #topics_in_doc = 10
        #topics_in_doc = int(len(line_parts) / 10) # 20 norm
        scores = [0 for _ in  range(topics_in_doc)] #RECENT
        #scores = [0 for _ in  range(10)]
        prev_score = 1
        if not len(line_parts) in [0, topics_in_doc*2]:
            line_parts = line_parts
        #assert len(line_parts) in [0, topics_in_doc*2]
        if not len(line_parts) % 2 == 0:
            line_parts = line_parts
        #assert len(line_parts) % 2 == 0
        if len(line_parts) < 2:
            DocMainTopic.append(-1)
            TopicScore.append(scores) #RECENT
            EmptyTopics +=1
        else:
             for i in range(len(line_parts) // 2):
            #for i in range(1): #C2 , while waiting from SS for all 25
            #for i in range(topics_in_doc): #RECENT
            #for i in range(10):
                topic_str = line_parts[2*i+1].replace('[','').replace('(','').replace(')','').replace(' ','').replace(']','').replace('\n','')
                topic_id = int(topic_str)
                score = float(line_parts[2*i].replace('[','').replace('(','').replace(')','').replace(' ',''))
                scores[topic_id] = score
                #scores[topic_id] += random.randint(0, 1000)/100000 - 0.005 # random noise
                assert score <= prev_score
                prev_score = score
                if i == 0:
                    DocMainTopic.append(topic_id)
                    TopicCounts[topic_id]+=1
             TopicScore.append(scores)
 print("Empty topics:", EmptyTopics)
 TopicWord = []
 for topic in range(topics_in_doc):
    #for topic in range(10):
        TopicWord.append(collections.OrderedDict())
    #print (genre, TopicCounts)
    #print (genre, TopicCounts, f"skipped: {skipped_empty}")
 #sys.exit(0) #

#Need to read each topic's words from a file
if UseTopics: #  ok to dis when no keywords are generated, C2: < 1'
    Lines = open("ukwac1c.25.topics.lst.txt").readlines()
    #Lines = open("ukwac1b.25.topics.lst.txt").readlines()
    #Lines = open("ukwac1a.25.topics.lst.txt").readlines()
    #Lines = open("genre1.10.topics.lst.txt").readlines()
    num_topics = 25 # before early july was 10
    assert len(Lines) == num_topics
    for topic_id in range(num_topics):
            line_parts = Lines[topic_id].split()
            #assert len(line_parts)/2 > 47 #
            assert len(line_parts)/2 > 4500
            line_parts = line_parts[:2000] #RECENT
            for j in range(int(len(line_parts)/2)):
                word = line_parts[2*j]
                score = float(line_parts[2*j+1])
                TopicWord[topic_id][word] = score


#Read top terms for the docs
def TopTerms(docid, words): #returns top most related to the doc topics words that are present in the doc
#C2: only needs 'TopicWord', so not using genre/topic Cache
    #ssert len(words) > 10 #RECENT # dis
    Scores = collections.OrderedDict()
    #For all words in the doc
    #for word in TopicWord[DocMainTopic[docid]]: #using only words from the dominant topic
    for word in words:
        #for topic in  [DocMainTopic[docid]]: #for generating: using only dominant topic #RECENT
        for topic in range(25): # #RECENT for genre training:  still using topic distribution to score the keywords
        #for topic in range(10):
            #topic_id = random.randint(0, 9) # this does not affect  match, so probably mid DF words are sufficient!
            topic_id = topic
            if word in TopicWord[topic_id]:
                score = TopicWord[topic_id][word] #RECENT regardless of LDA topic composition, but still promoting words that are common among LDA tops
                #score = TopicScore[docid][topic_id]*TopicWord[topic_id][word] #IDEA: try using const? or only TopicWord[topic_id][word] ?
                #score = TopicScore[docid][topic_id]*TopicWord[topic_id][word]
                #Obtain how much related to the doc:  TF*(sum over topics, topic_score*how_much_related_to_topic
                if word not in Scores:
                    Scores[word] = score
                else:
                    Scores[word] += score

    sorted_tuples = []
    for word in Scores:
                sorted_tuples.append((word, Scores[word]))

    #Sort by being topically related to the doc
    sorted_tuples = sorted(sorted_tuples, key=lambda student: student[1])
    # Truncate to top N  and convert to string
    #TopicWordsPreserve = 5000 #
    TopicWordsPreserve = args.tops #RECENT
    #TopicWordsPreserve = 40 #RECENT
    #TopicWordsPreserve = 40  # mostly used 20, then 40 # now when trying to ignore word presence in the doc
    #TopicWordsPreserve = 10 #tried, but did not reflect topics well, esp topics 3, 5, 7
    tops = [sorted_tuples[len(sorted_tuples)-i-1][0] for i in range(len(sorted_tuples))] #RECENT, SLOW?, inversing to compare visually
    #tops = [s for s,c in sorted_tuples[-2*TopicWordsPreserve:]]  ##-2* just incase off in the indexes? below takes cut anyway
    #tops = [s for s,c in sorted_tuples[-TopicWordsPreserve:]]
    #tops = [s for s,c in sorted_tuples[:TopicWordsPreserve]] #indeed more rare words sneak in
    #assert len(tops) >= TopicWordsPreserve
    out = []
    for word in words:
        if word in tops[:TopicWordsPreserve]:
            out.append(word)
    #random.shuffle(tops) #RECENT dis for combining LDA and "summarize:" result; RECENT trying  for arxiv, prior to that was disabled when low DF added back , disabled prior to that for for all 1000-256 tests. not doing for 512-5000. did for all :4000 gens for SS

    # assert False #did this mess up genre training files?: - No, all valid, just a bit diff, so will keep for a while
    return ' '.join(out) #
    #return str(DocMainTopic[docid]) + ' ' +  ' '.join(out) #
    #return str(DocMainTopic[docid]) + ' ' +  ' '.join(tops[:TopicWordsPreserve]) # back to norm, while "maintop--expand-1e" was with the above #
    #return ' '.join(tops[:TopicWordsPreserve]) # back to norm, while "maintop--expand-1e" was with the above
    #return ' '.join(tops)


def RemoveStopWords(words):
 out = []
 for w in words:
     if w not in Stops and random.randint(0,1) != 1:  #also added "dropping" random keywpods
         out.append(w)
 #return [w in words if w not in Stops]
 return out

def Keys(line): #returns the list of keywords for the line
    assert False #RECENT not used now so no point to maintain
    sorted_tuples = []
    for word in line:
    #for word in Clean(line).split():
    #for word in line.strip().split():
        if word in DFs:
            if DFs[word] > 10: #RECENT, but not tested much since quickly stopped using low DF keywords
                sorted_tuples.append((word, DFs[word]))
        else:
            assert False
    sorted_tuples = sorted(sorted_tuples, key=lambda student: student[1])  # sort by age
    Preserved = collections.OrderedDict()
    assert False #since:
    for word in [s for s,c in sorted_tuples[:20]]:  #RECENT what is 20 doing there? affected top40 and top100? - but this is not used now
        Preserved[word] = len(Preserved)
    out = []
    for word in line:
    #for word in Clean(line).split():
    #for word in line.replace("\"", ' ').replace(".", ' ').replace("\/", ' ').split():
        if word in Preserved:
            if random.randint(0,1) == 0:  #RECENT, but not tested much since quickly stopped using low DF keywords
                out.append(word)
    return out
    #return [s for s,c in sorted_tuples[:20] ]


if args.keywords_from_test:
    f = open(args.out, "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    for line in open(args.inp, encoding='utf-8').readlines()[1:]:
        part = line.split('\t')
        f.write(part[0] + '\t' + part[1] + part[2] + '\t' + TopTerms(int(part[0]), part[2].split()) + '\texpand\n')
        #f.write(part[0] + '\t' + part[2] + '\t' + TopTerms(args.topic - 1, Clean(part[2]).split()) + '\texpand\n')
    sys.exit(0)

if args.keywords_from_test_split:
    f = open(args.out, "w", encoding='utf-8')

    fg  = open(f"train-stories-{args.label}.txt", "w")
    fg.write("\ttarget_text\tinput_text\tprefix\n")
    fg.close()
    fg  =open(f"train-wiki-{args.label}.txt", "w")
    fg.write("\ttarget_text\tinput_text\tprefix\n")
    fg.close()
    fg  =open(f"train-hyper-{args.label}.txt", "w")
    fg.write("\ttarget_text\tinput_text\tprefix\n")
    fg.close()
    fg  = open(f"train-giga-{args.label}.txt", "w")
    fg.write("\ttarget_text\tinput_text\tprefix\n")
    fg.close()
    fg  = open(f"train-stack-{args.label}.txt", "w")
    fg.write("\ttarget_text\tinput_text\tprefix\n")
    fg.close()
    fg  = open(f"train-reviews-{args.label}.txt", "w")
    fg.write("\ttarget_text\tinput_text\tprefix\n")
    fg.close()

    for line in open(args.inp, encoding='utf-8').readlines()[1:]:
        part = line.split('\t')
        if part[1] == "__id__A11-icwsm09stories.real.ol":
            name = f"train-stories-{args.label}.txt"
        if part[1] == "__id__A16-wiki.ol":
            name = f"train-wiki-{args.label}.txt"
        if part[1] == "__id__A01-discuss-hyper-sample.ol":
            name = f"train-hyper-{args.label}.txt"
        if part[1] == "__id__A8-giga-en.clean.ol":
            name = f"train-giga-{args.label}.txt"
        if part[1] == "__id__A7-stackexchange-sample.ol":
            name = f"train-stack-{args.label}.txt"
        if part[1] == "__id__A17-review-sample.ol":
            name = f"train-reviews-{args.label}.txt"
        fg = open(name, "a")
        fg.write(part[0] + '\t' + part[2] + '\t' + TopTerms(int(part[0]), part[2].split()) + '\texpand\n')
        fg.close()
    sys.exit(0)


NotAllGenere = False

Genres = open("all-genre.txt").readlines()
Genres = [g.replace('\n','') for g in Genres]
#gi = 0
#genre = Genres[gi]
#genre = "__id__A17-review-sample.ol" #
#genre = "__id__A16-wiki.ol" #
#genre = "__id__A8-giga-en.clean.ol" #
#genre = "__id__A9-ukuslegislation.ol" #
#genre = "__id__A7-stackexchange-sample.ol"
#genre = "__id__A01-discuss-hyper-sample.ol"
#genre ="__id__A12-all"
#genre = "__id__A11-icwsm09stories.real.ol"
#genre = "__id__A4-brown-bnc-cen.ol" # for exploring pre-trained summarize:
#genre = "__id__arxiv.ol"
genre1 = "arxiv.ol"
#genre1 = "A14-arxiv.ol"
genre2 = "__id__A4-brown-bnc-cen.ol"
genre3 = "__id__A17-review-sample.ol"
genre4 = "A9-ukuslegislation.ol" #
#genre4 = "__id__A7-stackexchange-sample.ol"
#genre = "__id__A7-stackexchange-sample.ol"


#creating a balanced set of keywords for all topics and genre
#MaxToUse = 10000
#RECENT dis

if args.keywords_from_genre_test:
    TestIDs = collections.OrderedDict()
    for line in open("classifier/data/test-2000-no-punct.10k.tsv").readlines()[1:]:
    #for line in open("classifier/data/test-2000-no-punct.10k.tsv").readlines()[1:]:
        TestIDs[int(line.split('\t')[0])] = len(TestIDs)
if args.balanced_keywords or args.keywords_from_genre_test:
    MaxToUse = 10000000 #
    Lines = open("keywords-for-all-maintop-"+args.label+".txt", encoding='utf-8').readlines()[:MaxToUse] #RECENT RECENT added -2000
    assert False #C2 not using cache any longer
    DocGenreIDs  = open("genre-for-all-maintop-"+args.label+".txt").readlines()[:MaxToUse]
    DocTopicIDs  = open("topics-for-all-maintop-"+args.label+".txt").readlines()[:MaxToUse]
    #Lines = open("keywords-for-all-top100.txt", encoding='utf-8').readlines()[:MaxToUse] ##
    #DocGenreIDs  = open("genre-for-all-top100.txt").readlines()[:MaxToUse]
    #DocTopicIDs  = open("topics-for-all-top100.txt").readlines()[:MaxToUse]
    #Lines = open("keywords-for-all-top40.txt", encoding='utf-8').readlines()[:MaxToUse] ##
    #DocGenreIDs  = open("genre-for-all-top40.txt").readlines()[:MaxToUse]
    #DocTopicIDs  = open("topics-for-all-top40.txt").readlines()[:MaxToUse]
    #Lines = open("keywords-for-all-segs.txt", encoding='utf-8').readlines()[:MaxToUse] ##
    #DocGenreIDs  = open("genre-for-all-segs.txt").readlines()[:MaxToUse]
    #DocTopicIDs  = open("topics-for-all-segs.txt").readlines()[:MaxToUse]
    Sampled = numpy.zeros((10, 10))
    count = 0
    Genres = open("all-genre.txt").readlines()
    Genres = [g.replace('\n','') for g in Genres]
    f = open("keywords-from-test-"+args.label+".tsv", "w", encoding='utf-8')
    #f = open("test-tmp-" + str(seed_id) +  ".tsv", "w", encoding='utf-8')
    fl = open("labels-reviews-all-own-topics.txt", "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n")
    #get org sample for given genre
    #using each genre and each topic some number of times
    GenreIDToGen = 6 #brown - does not matter currently
    #for DocGenreID in range(10): # all of them
    for DocGenreID in [GenreIDToGen]: #5 for reviews, 4 for wiki #
    #for DocGenreID in range(10):
        for DocTopicID in range(25):#
        #for DocTopicID in range(10):
            #if DocTopicID in [9]: continue # excluding some for 25 topic LDA
            SameCount = 0
            for rep in range(3):# used so far mostly with 25 topics
            #for rep in range(10):#
                #find random doc DocGenreIDwith necessary topic
                DocId = -1
                Found = False
                if False:
                    #for _ in range(len(Lines)**2): #
                    for co in range(len(Lines)*20): #RECENT was *2 prior
                        DocId = random.randint(0, len(DocTopicIDs)-1)
                        if int(DocTopicIDs[DocId].replace('\n','')) == DocTopicID and int(DocGenreIDs[DocId].replace('\n','')) == GenreIDToGen: # to match both topic and genre
                        #if int(DocTopicIDs[DocId].replace('\n','')) == DocTopicID:
                            Found = True
                            break
                if not Found: #RECENT dis repeating among all genre
                    Found = Found
                    #SameCount += 1
                    for co in range(len(Lines)*2): #
                    #for co in range(len(Lines)*200): #RECENT was *2 prior
                        DocId = random.randint(0, len(DocTopicIDs)-1)
                        if args.keywords_from_genre_test and DocId+1 in TestIDs: #RECENT RECENT
                            Found = True #CARE: this does not depend on 'DocTopicID'
                            break
                        #if int(DocTopicIDs[DocId].replace('\n','')) == DocTopicID and int(DocGenreIDs[DocId].replace('\n','')) == GenreIDToGen: # to match both topic and genre
                        if Lines[DocId].split('\t')[0] != 'no summary produced': #RECENT
                         if int(DocTopicIDs[DocId].replace('\n','')) == DocTopicID:
                            Found = True
                            break
                    Found = Found
                Found = Found
                #assert Found # dis
                if DocId != -1:
                    #f.write(Lines[DocId].replace("summarize", "sep")) #if needed
                    SubstLine = Lines[DocId].split('\t')
                    #SubstLine[2] =  ' '.join(SubstLine[2].split(' ')[1:]) #
                    GenreStr = Genres[DocGenreID]
                    if DocGenreID == 3:
                        GenreStr = "arxiv.ol"

                    SameCount += 1

                    f.write(SubstLine[0] + '\t' + SubstLine[2] + '\t' + TopTerms(DocId, SubstLine[2].split() )+ '\tdummy\n') #RECENT, otherwise fields were not matching those in cache  ??
                    #f.write(SubstLine[0] + '\t' + SubstLine[1] + '\t' + TopTerms(DocId, SubstLine[1].split() )+ '\tdummy\n') # norm, bug with missing column fixed, fresh keywords, rather than from cache
                    #f.write(SubstLine[0] + '\t' + TopTerms(DocId, SubstLine[1].split() )+ '\tdummy\n') # RECENT fresh keywords, rather than from cache
                    #f.write(SubstLine[0] + '\t' + SubstLine[1] + '\t' + str(DocTopicID) + ' ' + TopTerms(DocId, SubstLine[1]) + '\tdummy\n') # fresh keywords, rather than from cache + topic id
                    #f.write(SubstLine[0] + '\t' + SubstLine[1].replace('"','') + '\t' + SubstLine[2].replace(Genres[int(DocGenreIDs[DocId].replace('\n',''))], Genres[DocGenreID] ) + '\tdummy\n') # only keywords?###RECENT, after not running for a while, not not entirely it was exactly the same code used for top 20 and earlier

                    #f.write(SubstLine[0] + '\tdummy\t' + SubstLine[1].replace('"','') +  '\tsummarize\n') # 2 segment input to summarize
                    #f.write(SubstLine[0] + '\tdummy\t' + SubstLine[2].replace('"','') +  '\tsummarize\n') # 2 segment input to summarize
                    #SubstLine[2] = labels[DocGenreID] + ' ' + ' '.join(SubstLine[2].split(' ')[2:]) # with summary
                    #SubstLine[2] = GenreStr + ' ' + ' '.join(SubstLine[2].split(' ')[2:]) # had to remove space to match train. not checked how it got there
                    #SubstLine[2] = ' '.join(SubstLine[2].split(' ')[1:]) #only keywords?
                    #SubstLine[2] = GenreStr + ' ' + ' '.join(SubstLine[2].split(' ')[1:]) #recent norm
                    #SubstLine[2] = Genres[DocGenreID] + ' ' + ' '.join(SubstLine[2].split(' ')[1:]) #olde tries
                    #f.write('\t'.join(SubstLine)) #norm
                    fl.write(str(DocGenreID) + '\t' + str(DocTopicID) + '\n')
            print(str(SameCount))
    quit()
#'''
'''
for line in Lines: #using original genre and topic

    #if not yet enough such genre-topic pairs, then add the line and save its genre/topic labels
    DocGenreID = int(DocGenreIDs[count].replace('\n',''))
    DocTopicID = int(DocTopicIDs[count].replace('\n',''))
    if DocGenreID == 4: #
     if Sampled[DocGenreID][DocTopicID] < 10: #
            f.write(Lines[count])
            Sampled[DocGenreID][DocTopicID] += 1
            fl.write(str(DocGenreID) + '\t' + str(DocTopicID) + '\n')


    count += 1
quit()
'''

if args.topic_stats:
    def id(line):
        return int(line.split('\t')[0])
    TopicCounts = [0 for _ in range(25)]  # RECENT
    GenreCounts = [0 for _ in range(10)]  # RECENT
    for line in open(args.inp, encoding='utf-8').readlines()[1:]:
        topic_id = DocMainTopic[id(line)] # RECENT
        TopicCounts[topic_id] += 1
        genre_str = line.split('\t')[1]
        if genre_str == '__id__A12-all':
            genre_str = '__id__A12-all.ol'
        GenreID = Genres.index(genre_str)  # RECENT #RECENT
        GenreCounts[GenreID] += 1
    print("TopicCounts:", TopicCounts)
    print("GenreCounts:", GenreCounts)
    sys.exit(0)


#Topics = open("genre.10.topics.lst.clean.txt", encoding='utf-8').readlines() #RECENT dis
MaxLineInGenreCorpus =  10000000 #C2
#MaxLineInGenreCorpus =  105438 #300 #  #105216 total, but some lines may not have labels so better use 105438+ # 30k was not enough to get reviews, but 100k was ok was legislation, but not sure for all of it
#MaxLine = 40000 + 100 #
MaxLine = args.genre_cap #C2
#MaxLine=MaxLineInGenreCorpus #300 #  #105216 total, but some lines may not have labels so better use 105438+ # 30k was not enough to get reviews, but 100k was ok was legislation, but not sure for all of it
#SampleEach = 100 #
#SampleEach = 100000 # RECENT for 25 topics - using large now to avoid capping #C2 dis
#SampleEach = 4000 # some topics have only ~4100
NotAllTopics = False
assert args.data == True #don't change from default, usage carried from older code RECENT C1+C2

if True: #C1+C2
#if not args.aug: #RECENT
    Name = args.out #C1+C2
    #Name = "classifier/data/train-single-topic-excluded-"+str(args.topic)+"-" + args.label+ ".tsv"  if not args.overall else "classifier/data/train-" + args.label+ ".tsv"
    f = open(Name, "w", encoding='utf-8')
    #f = open("classifier/data/train-single-topic-excluded-"+str(args.topic)+"-" + args.label+ ".tsv" , "w", encoding='utf-8') #RECENT
    #f = open("train-" + labels[gi] + "-topLDA.tsv" , "w", encoding='utf-8')
    #f = open("tmp-output-4000-top100.tsv" , "w", encoding='utf-8')
    #f = open("train-" + labels[gi] + "-maintop-500.tsv" , "w", encoding='utf-8')
    #f = open(partition + "-arxiv-300-topic-keywords-" + str(seed_id) + ".tsv", "w", encoding='utf-8')
    f.write("\ttarget_text\tinput_text\tprefix\n") # since will be concat-ing
count = 1
ReGeneratingCache = args.regen
assert args.regen == False #C2: no longe rneeded
if False:
#if ReGeneratingCache:
    fg  = open("genre-for-all-maintop-"+args.label+".txt", "w", encoding='utf-8') #RECENT
    #fg  = open("genre-for-all-maintop.txt", "w", encoding='utf-8')
    ft = open("topics-for-all-maintop-"+args.label+".txt", "w", encoding='utf-8') #RECENT
    #ft = open("topics-for-all-maintop.txt", "w", encoding='utf-8')
    fk = open("keywords-for-all-maintop-"+args.label+".txt", "w", encoding='utf-8') #RECENT
    #fg  = open("genre-for-all-top40.txt", "w", encoding='utf-8')
    #ft = open("topics-for-all-top40.txt", "w", encoding='utf-8')
    #fk = open("keywords-for-all-top40.txt", "w", encoding='utf-8')
    #fg  = open("genre-for-all-segs.txt", "w", encoding='utf-8')
    #ft = open("topics-for-all-segs.txt", "w", encoding='utf-8')
    #fk = open("keywords-for-all-segs.txt", "w", encoding='utf-8')
    fkl = []

Genres = open("all-genre.txt").readlines()
Genres = [g.replace('\n','') for g in Genres]

#TopicCount = [0 for _ in range(10)]


ExcludedIDs = collections.OrderedDict()
if args.ids_exclude != "":
    for line in open(args.ids_exclude,encoding='utf-8').readlines()[1:]:
        ExcludedIDs[int(line.split('\t')[0])] = len(ExcludedIDs)

for pa in range(2): #C2
 lines_topic_marked = 0
 training = []
 TopicCount = [0 for _ in range(25)]  #C2 moved here
 GenreCounts = [0 for _ in range(10)]
 #genre_index = 0 #C2
 # fC2 = open(f"C2.{genre_index}.txt", "w", encoding='utf-8') #C2

 #fr = open("fr-tmp.tsv", "w", encoding='utf-8') #
 #fr = open("2seg-to-summarize.tsv", "w", encoding='utf-8') #
 #fr2 = open("fr2.tsv", "w", encoding='utf-8') #
 #fr = open(labels[gi] + "-2seg-to-summarize.tsv", "w", encoding='utf-8') #
 #fr2 = open(labels[gi] + "-fr2.tsv", "w", encoding='utf-8') #
 #fr.write("\ttarget_text\tinput_text\tprefix\n")
 already_extracted = False
 sentences = []
 count_split = 0
 last_genre = ''
 genre_id = -1

    #to create cached versions: # - no longer needed as long as we don't shuffle docs
    #
    #assert ReGeneratingCache
    #assert False #CARE: since file with topics no longer matches - ok if not shuffling?
    #LinesShuf = open(Path, encoding='utf-8').readlines() # only for generating dataset for topic classification
    #random.shuffle(LinesShuf) # dis since need to refer to the summaries CARE: use if only sequential reading not expected
    #for line in LinesShuf:
    #'''lines_topic_marked
    #use # comment for below two lines if ReGeneratingCache
 #count_C2 = 0
 topic_scores = [[] for _ in range(10)]  # SLOW?
 too_small = [0 for _ in range(10)]
 for genre_index in IndexGenreUsed:
 #for genre_file in C2CorpusPaths:
  added_cur_genre = 0
  checked_cur_genre = 0
  for corpus in CorporaUsed:
  #for corpus in [1,2]:

    lines = 0
    if corpus == 2:
         Path = f"S:/genre-corpus2/{C2CorpusPaths[genre_index]}"  # C2
    else:
         Path = f"corpus{genre_index}.txt"  # C1+C2
    #all_lines = open(f"topics{gi}.txt").readlines()  # C2
    #Path = f"corpus{genre_index}.txt"  #C1+C2
    #Path = f"S:/genre-corpus2/{C2CorpusPaths[genre_index]}"  #C2
    for line in open(Path, encoding='utf-8'): #norm: sequential reading from file #older version: for line in open(Path, encoding='utf-8').readlines():
        #assert not ReGeneratingCache  # otherwise uncomment above code #RECENT dis
        lines += 1 #CARE: need this to cap each genre
        if lines > MaxLine:  #
            #continue #C2
            break #RECENT
        #if lines % EveryThLineUsed != 0: #RECENT using only every Xth line #RECENT moved below
        #    continue

        '''
        m = re.search("\[[^\]]+\]", line)
        genre_reg = r"__id__.+\.ol+[ \t]|[0-9]+.[0-9]+v[0-9]" #RECENT
        #genre_reg = r"__id__.+\.ol|[0-9]+.[0-9]+v[0-9]+"'''

        #assert re.search(genre_reg, line) != None #fails at a few samples in the  corpus with '\n' in them
        if True: #C2  # decided not to mess now with discarding short docs to avoid breaking consistency with the topic scores
        #if len(line) > 10: #C2
        #if re.search(genre_reg, line) != None: #RECENT
        #if m != None:
            #f.write(line[m.regs[0][0]:m.regs[0][1]] + '\n')
            lines_topic_marked += 1
            checked_cur_genre += 1
            if lines_topic_marked  % EveryThLineUsed != 0: #RECENT moved here and conditioned on lines_topic_marked
                continue
            '''if pa == 0:  # C2
                # later add collecting info for sorting docs within each genre
                continue'''

            rest_of_line = line
            if len(rest_of_line) < args.cut:  # C1+C2 moved here, before 'topic_scores' added
                # if len(rest_of_line) < args.random_cut: #C2
                too_small[genre_index] += 1  # RECENT was GenreID ???
                # too_small += 1
                continue

            #C1+C2: moved up so affects both passes #RECENT later also moved prior to topic_scores[genre_index].append(...
            if args.ids_exclude != "" and not args.inverse_exclusion and lines_topic_marked in ExcludedIDs:  # RECENT
                continue
            if args.ids_exclude != "" and args.inverse_exclusion and lines_topic_marked not in ExcludedIDs:  # RECENT
                continue

            topic_scores[genre_index].append(TopicScore[lines_topic_marked - 1][args.topic - 1])  # RECENT C2 moved here


            if pa == 0: #RECENT
                continue
            if corpus == 1 or genre_index not in [4, 7]: #C2
                rest_of_line = '\t'.join(line.split('\t')[1:])
            #rest_of_line = line[m.regs[0][1]:].replace('\t', ' ')
            #rest_of_line = rest_of_line[:1500] #RECENT
            #rest_of_line = rest_of_line[:1500] # to experiment with generating summaries, still trimmed by sentences
            if args.random_cut: #RECENT
                start_cut = random.randint(0, max(0, len(rest_of_line) - args.cut))
                rest_of_line = rest_of_line[start_cut:start_cut + args.cut]  # RECENT
            else:
                rest_of_line = rest_of_line[:args.cut]  # RECENT RECENT, 2000 used for early memory tests
            #rest_of_line = rest_of_line[:4000] #RECENT back to that #has done for weeks+  for SS request; back was  disabling this to prepare some cached "raw" extracts by genre, have only for arxiv

            #rest_of_line = rest_of_line[:1000] #all tests with low DF
            #rest_of_line = rest_of_line[:5000] #was used for the 512 single base model for all genre
            #rest_of_line = rest_of_line[:1000] #RECENT to speed things up, get a feel, used for all-genre, and prior
            #rest_of_line = re.sub(r"__id__[^ ]+ ", "", rest_of_line) #copied below

            #moved here from after 'if' below so all words are added to the dict

            # sentences.append(Clean(rest_of_line).split())  # RECENT dis: needed only for DF info

            genre_str  = Genres[genre_index] #C2
            GenreID = genre_index #C2
            #GenreID = Genres.index(genre_str) #RECENT
            '''if not args.data:  # creating test set for the excluded topic #C2 dis
                if GenreCounts[GenreID] >= args.cap:  #
                    continue'''
            if not lines_topic_marked - 1 < len(DocMainTopic):
                lines_topic_marked = lines_topic_marked
            assert  lines_topic_marked - 1 < len(DocMainTopic)
            topic_id = DocMainTopic[lines_topic_marked - 1] #RECENT

            if not ReGeneratingCache and not genre_str in ["__id__A01-discuss-hyper-sample.ol", "__id__A11-icwsm09stories.real.ol", #RECENT
                                 "__id__A16-wiki.ol", "__id__A17-review-sample.ol", #RECENT 09/07 excluded A12
                                 #"__id__A12-all.ol", "__id__A16-wiki.ol", "__id__A17-review-sample.ol",
                                 "__id__A7-stackexchange-sample.ol",
                                 "__id__A8-giga-en.clean.ol"]:  # norm for batch tests on topic exclusion #RECENT
                continue


            if args.by_topics:
             if not ReGeneratingCache and ((args.data and not args.overall and not (topic_id+1 != args.topic)) or (not args.data  and not (topic_id+1 == args.topic))):  # second clause to create keywords for gens for augmentation #RECENT norm
             #if not ReGeneratingCache and ((args.data and not args.overall and not (topic_id+1 not in [6])) or (not args.data  and not (topic_id+1 in [6]))):  #trying list, did not test
                continue #RECENT

            if args.num_top != 0 and pa == 1 and added_cur_genre >= args.num_top: #RECENT dis
                continue #C1+C2
            if args.num_bottom != 0 and pa == 1 and  added_cur_genre >= args.num_bottom:
                continue  # C1+C2

            assert  not args.overall #C1+C2
            assert not ReGeneratingCache  # C1+C2
            assert not args.by_topics # C1+C2

            assert not (args.num_bottom and args.num_top)
            if args.num_top != 0 and pa == 1:
             if not TopicScore[lines_topic_marked - 1][args.topic - 1] >= top_thresh[genre_index]:
                continue #C2
            if args.num_bottom != 0 and pa == 1:
             #if TopicScore[lines_topic_marked - 1][args.topic - 1]  > top_thresh[genre_index] or checked_cur_genre  < 1000: #to verify that can bring docs from mid as well
             if TopicScore[lines_topic_marked - 1][args.topic - 1] > top_thresh[genre_index]:
                continue #C2

        #if not ReGeneratingCache and (genre not in genre_str): # norm if only one genre used
            #if genre1 not in rest_of_line and genre2 not in rest_of_line and genre3 not in rest_of_line and genre4 not in rest_of_line and not (38641 < lines  <  49062): #when using only specific genre
            #if 38641 > lines  or lines > 49062: #for arxiv?
            #if False: #  norm when using all genre #, creating training keywords for all genre
            '''if not ReGeneratingCache and (genre not in rest_of_line): #when using only specific genre
                if already_extracted: # dis since working with several genre
                    print(rest_of_line)
                    break
                continue''' #RECENT dis
                #break #
            #fr2.write(line)  #  moved here
            rest_of_line = rest_of_line.replace(genre_str, '')  # RECENT moved here
            if True:  # norm

                tmp = []
                for c in rest_of_line.lower():# RECENT
                    #tmp.append(c if 'a' <= c <= 'z' or c in ['.', ',',';','-','?','!',"'",'"'] else ' ') #
                    if 'a' <= c <= 'z':
                        tmp.append(c)
                    #tmp.append(c if 'a' <= c <= 'z' else ' ') #did for experiments prior to 30/05/22
                    if args.punct: #
                        if c in [' ', '.', ',', ';', '-', '?', '!', "'",'"']:
                            tmp.append(c)
                    else:
                        if c in [' ','.', ',',';','-','?','!',"'",'"']: # since was not sure "st" tokenizer always detaches the punctiation
                            tmp.append(' ')
                            #tmp.append(' ');tmp.append(c);tmp.append(' ')
                rest_of_line = ''.join(tmp)


                #rest_of_line = rest_of_line[50:350] #RECENT - tried for arxiv, to the point then genre started to work on all topics, but decided too much of non-words so hard to read and assess
                #rest_of_line = rest_of_line.replace(genre_str, '') #RECENT moved here

                # For classification:
                topic_id = DocMainTopic[lines_topic_marked-1]
                assert 0 <= topic_id < 25 or topic_id == -1 #C2

                rest_of_line =  rest_of_line[5:] if rest_of_line[:5] == '  q h' else rest_of_line #for stack #RECENT
                while "h history h" in rest_of_line: #for wiki #RECENT
                    rest_of_line = rest_of_line.replace("h history h", " ")
                rest_of_line_list = rest_of_line.split()
                while 'h' in rest_of_line_list:
                    rest_of_line_list.remove('h')
                while 'p' in rest_of_line_list:
                    rest_of_line_list.remove('p')
                while 'b' in rest_of_line_list:
                    rest_of_line_list.remove('b')
                rest_of_line = ' '.join(rest_of_line_list)

                #RECENT 25/06, but after LM datafiles already generated, copied from from-SS-genre-to-t5.py:
                while "url afe" in rest_of_line:  #added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("url afe", " ")

                while "m-bm-m" in rest_of_line:  # added later after .more model was already trained
                        rest_of_line = rest_of_line.replace("m-bm-m", " ")
                while "m-cm-" in rest_of_line:  # added later after .more model was already trained
                        rest_of_line = rest_of_line.replace("m-cm-", " ")
                while "m-bm-" in rest_of_line:  # added later after .more model was already trained
                        rest_of_line = rest_of_line.replace("m-bm-", " ")

                while "bm-ys" in rest_of_line:  # added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("bm-ys", " ")
                while "-bm-s" in rest_of_line:  # added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("-bm-s", " ")
                while "-bm-y" in rest_of_line:  # added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("-bm-y", " ")
                while "-bm-" in rest_of_line:  # added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("-bm-", " ")




                rest_of_line = re.sub("url\"afe.*", '', rest_of_line) #RECENT
                rest_of_line = re.sub('doc url \".*', '', rest_of_line) #RECENT
                rest_of_line = re.sub('doc url.*', '', rest_of_line) #RECENT
                #rest_of_line = re.sub('doc url\".*', '', rest_of_line)
                #rest_of_line = re.sub('doc url\".*\"', '', rest_of_line)
                rest_of_line = re.sub('url \".* ', '', rest_of_line) #RECENT
                #rest_of_line = re.sub('url\".*', '', rest_of_line)
                if "like us on facebook" in rest_of_line:
                    rest_of_line = rest_of_line
                rest_of_line = re.sub('like us on facebook.*', ' ', rest_of_line)

                rest_of_line = re.sub('m-cm^[\ ].* ', ' ', rest_of_line) #RECENT #09/07 changed ALL "m-bm" patterns to 'm-bm[^ ].* '
                #rest_of_line = re.sub('m-cm.* ', ' ', rest_of_line)
                rest_of_line = re.sub('m-bm^[\ ].* ', ' ', rest_of_line)
                #rest_of_line = re.sub('m-bm.* ', ' ', rest_of_line)
                rest_of_line = re.sub("bamp\;q", ' ', rest_of_line)
                rest_of_line = re.sub("nbsp;", ' ', rest_of_line)
                if rest_of_line[-len("doc"):] == "doc":
                    rest_of_line = rest_of_line[:-len("doc")]

                while "nbsp" in rest_of_line:  # added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("nbsp", " ") #26/06, in case args.punct is False
                while "bamp q" in rest_of_line:  # added later after .more model was already trained
                    rest_of_line = rest_of_line.replace("bamp q", " ")  # 26/06, in case args.punct is False
                rest_of_line = re.sub('m cm^[\ ].* ', ' ', rest_of_line) #RECENT
                #rest_of_line = re.sub('m-cm.* ', ' ', rest_of_line)
                rest_of_line = re.sub('m bm^[\ ].* ', ' ', rest_of_line)


                rest_of_line = Clean(rest_of_line) #END copied



                if True: #C1+C2
                #if not args.aug: #RECENT, was below a recent mistake?:
                #if not args.aug and args.overall: #RECENT was next below wrong? but verified this created identical train file:  --cut 1000 --random --label 1000-r-tmp  --tops 10 --cap 100000 --overall --data  --ids_exclude "classifier/data/test-1000-r-excluding-by-topics-6,8,10,15,17,18-30.tsv"
                #if not args.aug and not args.overall: #RECENT RECENT
                    line_out = str(lines_topic_marked) + '\t' + genre_str.strip() + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ')   + '\t'+"genre"+'\n' #norm: to generate training data for genre classifier
                    # line_out = str(checked_cur_genre) + '\t' + genre_str.strip() + '\t' + rest_of_line.replace("\n",' ').replace("\t", ' ') + '\t' + "genre" + '\n'  #was temp to check index of selected docs inside each gene
                if args.extract_doc != -1:
                    if lines_topic_marked == args.extract_doc:
                        f = open("doc-exract-tmp.txt", "w")
                        f.write(line_out)
                        f.close()
                        exit(0)
                    continue

                assert not  args.topic_class #C1+C2
                if args.overall and args.topic_class: #RECENT
                    line_out = str(lines_topic_marked) + '\t' + str(topic_id+1)  + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ')   + '\t'+"topics"+'\n' # to generate training data for topics #RECENT dis
                #ine_out = str(lines_topic_marked) + '\t' + str(topic_id)  + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ')   + '\t'+"topics"+'\n' # to generate training data for topics #RECENT dis
                #line_out = str(lines_topic_marked) + '\t' + str(topic_id)  + '\t' + TopTerms(lines_topic_marked - 1, Clean(rest_of_line).split())   + '\t'+"topics"+'\n' # to use keywords only for topic classification,
                #line_out = str(lines_topic_marked) + '\t' + topic_str + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ')   + '\t'+"topics"+'\n' # to generate training data for topics #
                #line_out = str(lines_topic_marked) + '\t' + genre_str.strip() + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ')   + '\t'+"genre"+'\n' # # to generate training data for genre classifier
                #line_out = str(lines_topic_marked) + '\tdummy\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ')  +  '\tdummy\n' #only to create input to aply T5's "summarize:", prefix not important since hardcoded in the generator
                #line_out = str(lines_topic_marked) + '\t' + seg1 + ' .\t' + seg2 + ' .\tsegments\n'  # to save segments as keywords for creating a test set later
                #to creare cache to test expand from summaries+keywords #
                '''if genre_count < len(Summary[genre_id]):
                 line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' + labels[genre_id]   + ' ' + Summary[genre_id][genre_count] +  '\t'+prefix+'\n' #
                 assert genre_id ==  Genres.index(genre_str)
                 #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' + labels[genre_id]   + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split())  +  ' sep ' + Summary[genre_id][genre_count] +  '\t'+prefix+'\n' #
                else:
                 line_out = "no summary produced\t\t\t\n"'''

                # For genre LMs:
                '''if args.aug and not args.topic_class:
                    line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' +  TopTerms(lines_topic_marked-1, Clean(rest_of_line).split())  + '\texpand\n' #RECENT''' #C1+C2 dis
                #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' +  TopTerms(lines_topic_marked-1, Clean(rest_of_line).split())  + '\t'+prefix+'\n' #RECENT norm, when using LDA keywords, used for :4000 cut gens sent to SS #RECENT dis

                #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' +  str(DocMainTopic[lines_topic_marked-1]) + '\t'+prefix+'\n' #
                #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' + ' ' + genre_str +  ' ' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split())  + '\t'+prefix+'\n' # same as above + genre label in input
                #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' + genre_str +  ' ' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split()) + ' sep ' + ' '.join(Keys(Clean(rest_of_line.replace(genre,'')).replace('\n','').lower().split()))  + '\t'+prefix+'\n' #RECENT

                #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' + genre_str +  ' ' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split()) + ' ' + ' '.join(Keys(Clean(rest_of_line.replace(genre,'')).replace('\n','').lower().split()))  + '\t'+prefix+'\n' #RECENT both topic words and low df words#
                #line_out = str(lines_topic_marked) + '\t' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split()) + ' ' + ' '.join(Keys(Clean(rest_of_line.replace(genre,'')).replace('\n','').lower().split()))  + '\t' + str(topic_id) + '\t'+prefix+'\n' #only for keywords
                #line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' + genre_str +  ' ' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split()) +   '\t'+prefix+'\n' #RECENT removed Clean-ing the target text
                #line_out = str(lines) + '\n' #
                #line_out = str(lines_topic_marked) + '\t' + Clean(rest_of_line) + '\t' + genre_str +  ' ' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split()) +   '\t'+prefix+'\n' #RECENT keywords = top N words in the document as related to the doc topic # used that for stack-exchange and reviews, it modifies the formatting in the target text
                #line_out = str(lines_topic_marked) + '\t' + rest_of_line + '\t' + genre_str +  ' ' + TopTerms(lines_topic_marked-1, Clean(rest_of_line).split()) +   '\t'+prefix+'\n' #RECENT keywords = top N words in the document as related to the doc topic
                assert line_out.count('\t') == 3
                assert line_out.count('\n') == 1
                assert line_out.count('\r') == 0
                #line_out = str(count) + '\t' + text_without_genre_label + '\t' + str(topic_id) + ' ' + genre_str + '\t'+prefix+'\n' # with topic ID
                #line_out = str(count) + rest_of_line[0:50]  + '\n' #
                #line_out = str(count) + '\t' + ' '.join(text_copy_shuf) + '\t' + str(topic_id)  + '\t'+prefix+'\n' #generating keywords for given topic
                #line_out = str(count) + '\t' + text_without_genre_label + '\t' + ' '.join(text_copy_shuf)  + '\t'+prefix+'\n'
                #line_out = str(count) + '\t' + text_without_genre_label + '\t' + ' '.join(text_copy_shuf)  + '\t'+prefix+'\n'
                #assert line_out.count('\t') == 3 # dis
                #if True: # norm
                TopicCount[topic_id] += 1
                added_cur_genre += 1
                #topic_scores[genre_index].append(TopicScore[lines_topic_marked - 1][args.topic - 1]) #RECENT C2 moved here
                training.append(line_out) #RECENT all words shuffled as input , without commas and stopwords
                #training.append(line_out[:30] + '\n') #RECENT all words shuffled as input , without commas and stopwords
                #training.append(str(count) + '\t' + text_without_genre_label + '\t' + ' , '.join(text_copy_shuf)  + '\t'+prefix+'\n') # all words shuffled as input
                #training.append(str(count) + '\t' + rest_of_line.replace(genre,'').replace('\n','') + '\t' + ' '.join(Topics[topic_id].replace('\n','').split()) + ' ' + ', '.join(keywords)  + '\t'+prefix+'\n') # both topic words and low DF keywords
                #training.append(str(lines) + '\t'  +  ', '.join(keywords)   + '\t' +  ' '.join(Topics[topic_id].replace('\n','').split()) + '\t' + rest_of_line.replace(genre,'').replace('\n','')+ '\t'+prefix+'\n') #also prints genre label
                #training.append(str(lines) + '\t'  +  ', '.join(keywords)   + '\t' +  ' '.join(Topics[topic_id].replace('\n','').split()) + '\t'+prefix+'\n') #recent norm, for low DF keywords
                #training.append(str(count) + '\t' + rest_of_line.replace(genre,'').replace('\n','') + '\t' +  ', '.join(keywords)  + '\t'+prefix+'\n') #recent norm, for low DF keywords

                #training.append(str(count) + '\t' + rest_of_line.replace(genre,'').replace('\n','') + '\t' +  ' '.join(Topics[topic_id].replace('\n','').split())  + '\t'+prefix+'\n') # back to all topics
                #training.append(str(count) + '\t' + rest_of_line.replace(genre,'').replace('\n','') + '\t' +  ' '.join(Topics[topic_id].replace('\n','').split()[:5])  + '\t'+prefix+'\n')
                #training.append(str(count) + '\t' + rest_of_line.replace(genre,'').replace('\n','') + '\t' +  ' '.join(Topics[topic_id].replace('\n','').split()[:5])  + '\trelations\n')
                #training.append(str(count) + '\t' + rest_of_line.replace('__id__A01-discuss-hyper-sample.ol','').replace('\n','') + '\t' + Topics[topic_id].replace('\n','')  + '\trelations\n')
                #f.write(str(count) + '\t' + rest_of_line.replace('__id__A01-discuss-hyper-sample.ol','').replace('\n','') + '\t' + Topics[topic_id].replace('\n','')  + '\trelations\n') #CARE: may later need to change 'relations' to more accurate description
                # f.write(line) # just to see the lines
                #f.write(str(topic_id) + ' ' + topic_marker + ' ' + rest_of_line ) #for debugging
                count += 1
                already_extracted = True

                GenreID = Genres.index(genre_str) #RECENT
                GenreCounts[GenreID] += 1


                if ReGeneratingCache:
                    GenreID = Genres.index(genre_str)
                    fg.write(str(GenreID) + '\n')
                    ft.write(str(topic_id) + '\n')
                    fkl.append(line_out)
                    #fk.write(line_out)
    print(" ", lines-1,)
    #'''
 assert len(DocMainTopic) == len(TopicScore)
 assert lines_topic_marked == len(DocMainTopic) or EveryThLineUsed > 1 #C2
 if pa == 0 and (args.num_top != 0 or args.num_bottom != 0): #RECENT fixing bug below on 23/08
 #if pa == 0 and (args.num_top != 0 or args.num_bottom) != 0:
    top_thresh = [-1. for _ in range(10)]
    for i in range(10):
        topic_scores[i].sort()
        #tops_number = args.num_top if args.data else args.num_bottom
        if topic_scores[i] != []:
            if args.num_top: #test data:
                assert args.num_bottom == 0
                assert args.num_top+args.top_topic_reserve < len(topic_scores[i])
                top_thresh[i] = topic_scores[i][-(args.num_top+args.top_topic_reserve)] if topic_scores[i] != [] else 1.
                #top_thresh.append()
                #top_thresh.append(topic_scores[i][-(args.num_top)] if topic_scores[i] != [] else 1.)
            else: #train data:
                args.num_top == 0
                assert args.num_bottom + args.top_topic_reserve < len(topic_scores[i])
                top_thresh[i] = topic_scores[i][args.num_bottom-1] if topic_scores[i] != [] else 0.
                #top_thresh.append()
                #top_thresh.append(topic_scores[i][min(args.num_bottom, len(top_thresh)-args.reserve)-1] if topic_scores[i] != [] else 0.)
                #top_thresh.append(topic_scores[i][args.num_bottom-1] if topic_scores[i] != [] else 0.)
    #assert  args.num_top <= lines_topic_marked
    print("thresholds:", top_thresh)
 if pa == 1:
    print(TopicCount) #
    print(GenreCounts) ##RECENT
    print("discarded as too small:", too_small) #RECENT

#assert lines_topic_marked == min(len(DocMainTopic), MaxLine) or EveryThLineUsed > 1 # or NotAllTopics or NotAllGenere # dis
#assert lines_topic_marked == min(105216, MaxLine) or not ReGeneratingCache or EveryThLineUsed > 1 # or NotAllTopics or NotAllGenere # dis

#random.shuffle(fkl) #  only to inspect a random sample. CARE: this will violate consistency between other cache components>
if ReGeneratingCache:
    for line in fkl: # [1000:]: #
        fk.write(line)

if not args.no_shuf:
    random.shuffle(training) # RECENT undis

WantInTest = 0 #RECENT C2 CARE: changing this may affect the train for genre classifier and gens, so better not to until new round of experiments
    #I am not using the test sets created with WantInTest > 0 anyway
#WantInTest = 1000 if args.data and not args.aug and not args.overall_genre else 0 # for keywords only # 300 for 'nature' #  norm 1000 #for data to train gens #1000 for genre classifier #RECENT
#WantInTest = min(WantInTest, args.num_top) #C2  dis, leads to problems when args.data. not needed if I don't care for a few extra ones?
'''if args.overall_genre: #RECENT HACK
    WantInTest+= 1''' #C2 dis
#WantInTest = 1000 if args.data and not args.aug else 0 # for keywords only # 300 for 'nature' #  norm 1000 #for data to train gens #1000 for genre classifier #RECENT
assert not args.overall and not  args.overall_genre #C1+C2
if args.aug:  #RECENT norm RECENT
#if args.data and (not args.overall or args.overall_genre):  #RECENT norm RECENT
#if args.data and not args.overall:  # norm RECENT
    # if not args.aug:
    fg = open("train-giga-" + str(args.topic) + "-" + args.label + ".txt", "w",
              encoding='utf-8')  # added "-pr" for punctuation
    fh = open("train-hyper-" + str(args.topic) + "-" + args.label + ".txt", "w", encoding='utf-8')

    fs = open("train-stories-" + str(args.topic) + "-" + args.label + ".txt", "w", encoding='utf-8')
    fp = open("train-products-" + str(args.topic) + "-" + args.label + ".txt", "w", encoding='utf-8')
    fw = open("train-wiki-" + str(args.topic) + "-" + args.label + ".txt", "w", encoding='utf-8')
    fr = open("train-reviews-" + str(args.topic) + "-" + args.label + ".txt", "w", encoding='utf-8')
    ft = open("train-stack-" + str(args.topic) + "-" + args.label + ".txt", "w", encoding='utf-8')

    fg.write("\ttarget_text\tinput_text\tprefix\n")  #
    fh.write("\ttarget_text\tinput_text\tprefix\n")  #

    fs.write("\ttarget_text\tinput_text\tprefix\n")  #
    fp.write("\ttarget_text\tinput_text\tprefix\n")  #
    fw.write("\ttarget_text\tinput_text\tprefix\n")  #
    fr.write("\ttarget_text\tinput_text\tprefix\n")  #
    ft.write("\ttarget_text\tinput_text\tprefix\n")  #

if True: #C1+C2
#if not args.aug: #RECENT

  training_lines = training #C1+C2
  #training_lines = training if WantInTest == 0 else training[:-WantInTest] #C2 not sure why it worked without that prior
  '''if not args.data: #C2  HACK, should work for creating gen train data, later can figure for others
      training_lines = []'''
  for line in training_lines:  # train for genre classification #, for genre classification testing
  #for line in training[:-WantInTest]:  # train for genre classification #, for genre classification testing
  #for line in training[:-160]: # train for genre classification #TES, for genre classification testing
  #for line in training[:MaxLineInGenreCorpus]: #norm
    if True: #C1+C2
    #if not args.aug:
        f.write(line)
    if args.aug: #C1+C2
    #if not args.overall or args.overall_genre:
    #if not args.overall:
        parts = line.split('\t')
        lines_topic_marked = int(parts[0])
        rest_of_line = parts[2]
        line_out = str(lines_topic_marked) + '\t' + rest_of_line.replace("\n", ' ').replace("\t", ' ') + '\t' +  TopTerms(lines_topic_marked-1, Clean(rest_of_line).split())  + '\texpand\n' # norm, when using LDA keywords, used for :4000 cut gens sent to SS # RECENT undis
        if parts[1] == "__id__A01-discuss-hyper-sample.ol":
            fh.write(line_out)
        elif parts[1] == "__id__A11-icwsm09stories.real.ol":
            fs.write(line_out)
        elif parts[1] == "__id__A12-all.ol":
            fp.write(line_out)
        elif parts[1] == "__id__A16-wiki.ol":
            fw.write(line_out)
        elif parts[1] == "__id__A17-review-sample.ol":
            fr.write(line_out)
        elif parts[1] == "__id__A7-stackexchange-sample.ol":
            ft.write(line_out)
        elif parts[1] == "__id__A8-giga-en.clean.ol":
            fg.write(line_out)
        else:
            assert False

#OutFileName = arg.out #C1+C2
#TestFileName = "classifier/data/test-single-topic-excluded-"+str(args.topic)+"-" + args.label+ ".tsv"  if args.data else "classifier/data/test-single-topic-"+str(args.topic)+"-" + args.label+ ".tsv" #RECENT
'''if args.aug:
        TestFileName = "aug-keywords-"+str(args.topic)+"-"+args.label+".tsv" # 11/06 #C2'''
        #TestFileName = "aug-keywords-"+str(args.topic)+"-"+args.label+"-"+str(args.cap)+".tsv" # 11/06
if args.overall:
    assert False #C1+C2
    TestFileName = "classifier/data/test-" + args.label + ".tsv"

'''fte = open(TestFileName, "w", encoding='utf-8') #C1+C2 dis
#fte = open(TestFileName, "w", encoding='utf-8') #RECENT
#fte = open("genre-classifier-test.txt", "w", encoding='utf-8')
fte.write("\ttarget_text\tinput_text\tprefix\n") #RECENT
for line in training[-WantInTest:]:  #RECENT
#for line in training[:1000]: # test set for genre classification CARE: need to split above from train
    fte.write(line)'''
#print (len(training[:MaxLineInGenreCorpus]))

#for i in range(10):
#    assert TopicCount[i] == SampleEach

'''
ft = open("topics-test.txt", "w", encoding='utf-8')
ft.write("\ttarget_text\tinput_text\tprefix\n")
count = 1
max_topics = 100
for topic in Topics:
        #prefix  = "summarize"
        ft.write(str(count) +  '\tdummy\t' +  ' '.join(topic.replace('\n','').split()[:max_topics])  + '\t'+prefix+'\n')
        count += 1
'''


''' #to extract heading from <h>...</h>
Path ="A7-stackexchange-sample.ol"
#Path = "A9-ukuslegislation.ol" #headings too long
partition = "train"
f = open(partition + "-stackexchange-generate-t5.tsv", "w", encoding='utf-8')
f.write("\ttarget_text\tinput_text\tprefix\n")
count = 1

for line in open(Path, encoding='utf-8').readlines():
    m = re.search("<h>.*</h>", line)
    if m != None:
     if len(m.string) > 10:
        heading = line[m.regs[0][0]:m.regs[0][1]].replace('<h>', "").replace('</h>', "").strip()
        #line_parts = line.strip().split(['<h>', '</h>'])
        f.write(str(count) + '\t' + line[m.regs[0][1]:].replace('\t', "").replace('\n', "") + '\t' +  heading.replace('\t', "").replace('\n', "")  + '\trelations\n') #CARE: may later need to change 'relations' to more accurate description
        count += 1
'''