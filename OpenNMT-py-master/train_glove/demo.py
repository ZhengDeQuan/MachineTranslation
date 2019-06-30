from __future__ import print_function
import argparse
import pprint
# import gensim
from glove import Glove
from glove import Corpus

sentense = [['你','是','谁'],['我','是','中国人']]
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print(corpus_model.dictionary)
print('Collocations: %s' % corpus_model.matrix.nnz)
print(corpus_model.matrix)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
glove.save('glove.model')
glove = Glove.load('glove.model')
corpus_model.save('corpus.model')
corpus_model = Corpus.load('corpus.model')
print("most similar to 我")
print(glove.most_similar('我', number=10))
# 全部词向量矩阵
print(glove.word_vectors)
# 指定词条词向量

print("你")
print(glove.word_vectors[glove.dictionary['你']])
print(len(glove.word_vectors[glove.dictionary['你']]))
print(type(glove.dictionary))
for key in glove.dictionary.keys():
    print("key = ",key)
print(corpus_model.matrix.todense().tolist())