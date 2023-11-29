**The code and data for "BERT Goes Off-Topic: Investigating the Domain Transfer Challenge in Genre Classification" by Dmitri Roussinov, Serge Sharoff,  EMNLP Findings, 2023**

Here, you can find all the corpus and source files to re-create our experiments. Also, if running into difficulties re-creating the data sets, there are the datasets uploaded here for topic 1 so you can simply verify the gap between using on-topic and off-topic documents to train a genre classifier. This is the **main result** reported in our paper.

**1. Creating on-topic and off-topic datasets.**
   
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

To create on-topic training documents, you can use the following command:
```bash
python process-both.py --topic 1  --cut 1000 --random --label C1C2-30  --punct --tops 10   --genre_cap 10000000 --num_top 130 --ids_exclude classifier/data/test-single-topic-1-C1C2-1k-100-nop.tsv --out tmp-top.tsv
```
Additional arguments:
- `--num_top` (integer): how many on-topic documents per genre to extract
- `--ids_exclude` (string): the test file provides document IDs that needs to be excluded from training and testing sets to avoid overlaps.
  
After that,  similar to the above, you can run the following commands to remove punctuation and split into training and validation files called "train-top-1-C1C2-30-10kw.tsv" and "val-top-1-C1C2-30-10kw.tsv" accordingly:
```bash
python combine-aug.py --strip_punct --inp tmp-top.tsv  --out tmp-top-nop.tsv
python combine-aug.py --test_set classifier/data/test-single-topic-1-C1C2-1k-100-nop.tsv --inp tmp-top-nop.tsv
python combine-aug.py --reduce_to 30 --test_set classifier/data/val-top-1-C1C2-30-10kw.tsv  --out classifier/data/train-top-1-C1C2-30-10kw.tsv --inp tmp-top-nop.tsv
```
The training and testing files are also uploaded to the "classifier/data" folder for comparison.

If you want to re-create the test sets, you can use the following command:
```bash
python process-both.py --topic 1  --cut 1000 --random --label C1C2-1k  --punct --tops 10   --genre_cap 10000000 --num_top 130 --out tmp-top.tsv
```
then remove the punctuation and split into testig and validation sets (each 100 per genre) as shown above.

**2. Comparing on-topic and off-topic training.**

To test a Roberta-based classifier with an off-topic training set, the following command can be used:
```bash
python BertTrainerRoberta.py --log_label sample-30-classifier-roberta-large  --warmup_steps 160  --input_test data/test-single-topic-1-C1C2-1k-100-nop.tsv --input data/train-bottom-1-C1C2-30-10kw.tsv  --bert_model roberta-large --per_device_train_batch_size 1 --output_dir topic-1-C1C2-30-roberta-large   --logging_steps 96 --gradient_accumulation_steps 15 --seg_size 256  --num_train_epochs 36  --learning_rate 1e-05 --input_val data/val-bottom-1-C1C2-30-10kw.tsv --auto_val
```
Additional (not explained above) arguments:
- `--log_label` (string): the name of the log file with additional details.
- `--warmup_steps` (integer): number of warm up steps when training, simply passed to a Hugging Face function call.
- `--input_test` (string): the name of the testing set file
- `--input` (string): the name of the training set file
- `--bert_model` (string): this can be used to specify to use a base version, simply passed to a Hugging Face function call.
- `--per_device_train_batch_size` (integer): simply passed to a Hugging Face function call
- `--output_dir` (string): the name of the folder with the saved models.
- `--logging_steps` (integer): how often to test the model.
- `--gradient_accumulation_steps` (integer): simply passed to a Hugging Face function call
- `--seg_size` (integer):  max number of tokens in the input to the transformer model
- `--num_train_epochs` (integer):  how many epochs to train (there is no early stopping)
- `--learning_rate` (float):  simply passed to a Hugging Face function call
- `--input_val` (string): the name of the validatio set file.
- `--bert_model` (auto_val): wether to report F1 on the test set corresponding to the highest F1 on the validation set
The max validation F1 score reported when running this command should be around .93 +- .02 depending on the seed and number of shuffles. This corresponds to F1 on the test set around .49 +- 0.02.

To test a Roberta-based classifier with an on-topic training set, the following command can be used:
```bash
python BertTrainerRoberta.py --log_label sample-30-classifier-roberta-large-on-topic  --warmup_steps 160  --input_test data/test-single-topic-1-C1C2-1k-100-nop.tsv --input data/train-top-1-C1C2-30-10kw.tsv  --bert_model roberta-large --per_device_train_batch_size 1 --output_dir topic-1-C1C2-30-roberta-large   --logging_steps 96 --gradient_accumulation_steps 15 --seg_size 256  --num_train_epochs 36  --learning_rate 1e-05 --input_val data/val-top-1-C1C2-30-10kw.tsv --auto_val
```
The max validation F1 score reported when running this command should be around .92 +- .02. This corresponds to F1 on the test set around .89 +- 0.02. Thus, for this and other topics, the on-topic training results are better than off-topic training results
by about 25 absolute percentage points on average. This is the main finding in our paper.


