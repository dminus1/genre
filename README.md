# genre
The code and data for "BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff,  EMNLP Findings, 2023

1. Creating on-topic and off-topic datasets.
   
You can download all the necessary files [here](https://drive.google.com/file/d/1SsJRhy-TYtPBr_pp5JGOoWJKlO3KNRH9/view?usp=sharing) and [here](https://drive.google.com/file/d/1-EAkayPfV0upzEU09dmtmQbawN8VQ1gL/view?usp=sharing). They include slightly re-formatted corpora, the topic models, and other auxiliary files used by the Python code. Change the path from "S:" in the Python code to your downloaded location.

In order to create a dataset with a specificed number of  off-topic documents, run the following command:
```bash
python process-both.py --topic 1  --cut 1000 --random --label C1C2-30-0kw  --punct --tops 10   --genre_cap 10000000 --num_bottom 130 --top_topic_reserve 500 --out tmp-aug.tsv --aug
```
This should create the file "tmp-aug.tsv" uploaded here for comparison. The script accepts the following arguments:

- `--topic` (integer): which topic. E.g. for topic 1 ("entertainment"), the command will extract the LEAST related to entertainment documents
- `--cut` (integer): number of characters in the extracted document window.
- `--random` (boolean): whether the document window is randomly positioned
- `--punct` (boolean): whether to preserve punctuation
- `--tops` (integer): number of keywords to define the topic
- `--genre_cap` (integer): max number of documents per genre
- `--num_bottom` (integer): number of documents per genre to exrtact (e.g. here 100 for validation and 30 for training)
- `--top_topic_reserve` (integer): this is not important in the current version
- `--out` (string): the name of the output file that contains the documents for the dataset. Note that the program also creates training files for generating synthetic documents in the specified genre and on the specified topic (topic 1 in this example), with the file names for them here being train-giga-1-C1C2-30-10kw.txt, train-hyper-1-C1C2-30-10kw.txt, train-reviews-1-C1C2-30-10kw.txt etc. "giga" here stands for genre (news). "1"  stands for the topic, and the rest is the label defined by the next argument below.
- `--label` (string): the label to add to the file names mentioned right above, here C1C2 suggests that both corpus 1 and corpus 2 have been used,  the training size is 30 documents per genre, and the topic is represented by 10 keywords.
- `--aug` (boolean): whether to create the training files mentioned right above

There are a few more commands to futher re-organize training, validation and testing files. This one simply removes punctuation:
```bash
python combine-aug.py --strip_punct --inp tmp-aug.tsv  --out tmp-bottom-nop.tsv
```
The output "tmp-bottom-nop.tsv" is uploaded for comparison.

We are using the same test sets (100 documents per genre) for all sample sizes in the paper (30, 100 and 1000) so we can directly compare the performance. Those test sets should be extracted from the archive "test-sets.zip" and placed into the folder "classifier/data". The commands to re-create obtaining them are below. This command simply verifies that the test set does not overlap with the training and validation sets that are currently contained in tmp-bottom-nop.tsv:
```bash
python combine-aug.py --test_set classifier/data/test-single-topic-1-C1C2-1k-100-nop.tsv --inp tmp-bottom-nop.tsv
```
This command splits "tmp-bottom-nop.tsv" into training and validation sets as files "train-bottom-1-C1C2-30-10kw.tsv" and "val-bottom-1-C1C2-30-10kw.tsv" in the folder "classifier/data" (also uploaded here for comparison):
```bash
python combine-aug.py --reduce_to 30 --test_set classifier/data/val-bottom-1-C1C2-30-10kw.tsv  --out classifier/data/train-bottom-1-C1C2-30-10kw.tsv --inp tmp-bottom-nop.tsv
```
The script accepts the following arguments:
- `--reduce_to` (integer): number of training documents per genre
- `--out` (string): training set file name to create
- `--inp` (string): input dataset
- `--inp` (test_set): testing set name to verify that it does not overlap with training or validation sets


  


