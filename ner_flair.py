from typing import List
import gensim
import getopt
import json
import nltk
import os
import regex
import sys

from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import CharacterEmbeddings, CharLMEmbeddings
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.models import SequenceTagger
from flair.optim import SGDW
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

from torch.optim.adam import Adam

from hyperopt import hp

from nltk import tokenize
nltk.download('punkt')


print("\n---------- NER Module ----------\n")
columns = {0: 'token', 1: 'pos', 2: 'sublabel', 3: 'label'}
data_folder = "data/"

corpus: Corpus = ColumnCorpus(data_folder, columns, train_file="train_selective.txt",
                              test_file="test_selective.txt", dev_file="dev_selective.txt")

# Returns the number of items in an object.
print("\n Train length", len(corpus.train))
print("\n Test length", len(corpus.test))
print("\n Dev length", len(corpus.dev))

print(" ")
print("Train: ", corpus.train[0].to_tagged_string('label'))
print("Test: ", corpus.test[0].to_tagged_string('label'))
print("Dev: ", corpus.dev[0].to_tagged_string('label'))

tag_type = 'label'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

print(" ")
print("Tags: ")
print(tag_dictionary.idx2item)
print(" ")

# Loading NILC Word Embedding (Word2Vec_Skip-Gram_300d)
nilc_vectors = gensim.models.KeyedVectors.load_word2vec_format(
    "/content/drive/My Drive/iCybersec/data/bbp_word2vec_skpg_300d.txt")
nilc_vectors.save('nilc.gensim')

nilc_embedding = WordEmbeddings('nilc.gensim')

# Loading Flair Embedding
flair_embedding_forward = FlairEmbeddings(
    "/content/drive/My Drive/iCybersec/flairBBP_forward-pt.pt")
flair_embedding_backward = FlairEmbeddings(
    "/content/drive/My Drive/iCybersec/flairBBP_backward-pt.pt")

embedding_types: List[TokenEmbeddings] = [
    nilc_embedding,
    flair_embedding_forward,
    flair_embedding_backward,
]

embeddings: StackedEmbeddings = StackedEmbeddings(
    embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=SGDW)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.9,
              mini_batch_size=32,
              max_epochs=3,
              checkpoint=True)

#plotter = Plotter()
# plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
# plotter.plot_weights('resources/taggers/example-ner/weights.txt')

#path_eval = 'resources/taggers/example-ner/test.tsv'

#new_file = open("conlleval_test.tsv", "w+", encoding="utf8")

######
# The output of the test dataset have four elements by line: 'token tag1 tag2 score'.
# We need only 'token tag1 tag2' for CoNLL-2002 Script.
######

with open(path_eval, "r", encoding="utf8") as file:
    for line in file:
        if line != "\n":
            line = line.strip()
            spliter = line.split(" ")
            token = spliter[0]
            tag_1 = spliter[1]
            tag_2 = spliter[2]
            new_file.write(str(token)+" "+str(tag_1)+" "+str(tag_2)+"\n")
        else:
            new_file.write(line)
new_file.close()

path_final = "conlleval_test.tsv"

print(" ")
print("--- CoNLL-02 METRICS EVALUATION ---")
print(" ")

os.system("perl conlleval_02.pl < %s" % (path_final))
