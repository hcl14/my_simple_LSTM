# -*- coding: utf-8 -*-
from __future__ import division, print_function
#import gensim
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout, Reshape, Flatten, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from sklearn.cross_validation import train_test_split
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras import backend as K
from keras.objectives import cosine_proximity

import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import kaggle

import pickle

# set your paths to files
DATA_DIR = "data/comp_data"
MODEL_DIR = "data/models"
WORD2VEC_BIN = "/home/hcl/Documents/work/keyword-algorithms/process_reddit/model2/reddit_w2v_normalized.pickle"
WORD2VEC_EMBED_SIZE = 300

QA_TRAIN_FILE = "train.json"#"studystack_qa_cleaner_no_qm.txt"
QA_TRAIN_FILE2 = "test.json"
QA_TRAIN_FILE3 = "valid.json"


QA_EMBED_SIZE = 64
BATCH_SIZE = 32
NBR_EPOCHS = 2

## extract data

print("Loading and formatting data...")


# uncomment to prosess data
'''
qapairs1 = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE))

qapairs2 = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE2))
qapairs3 = kaggle.get_question_answer_pairs(
    os.path.join(DATA_DIR, QA_TRAIN_FILE3))

qapairs = qapairs1+qapairs2+qapairs3

with open("processed_input.pickle", "wb") as outfile:
    pickle.dump(qapairs, outfile,protocol=pickle.HIGHEST_PROTOCOL) 
'''

# loading preprocessed (tokenized) dataset
with open("processed_input.pickle", 'rb') as f:
    qapairs = pickle.load(f)


question_maxlen = max([len(qapair[0]) for qapair in qapairs])
answer_maxlen = max([len(qapair[1]) for qapair in qapairs])
seq_maxlen = max([question_maxlen, answer_maxlen])

word2idx = kaggle.build_vocab([], qapairs, [])
vocab_size = len(word2idx) + 1 # include mask character 0

Xq, Xa, Y = kaggle.vectorize_qapairs(qapairs, word2idx, seq_maxlen)
Xqtrain, Xqtest, Xatrain, Xatest, Ytrain, Ytest = \
    train_test_split(Xq, Xa, Y, test_size=0.3, random_state=42)
print(Xqtrain.shape, Xqtest.shape, Xatrain.shape, Xatest.shape, 
      Ytrain.shape, Ytest.shape)

# get embeddings from word2vec
# see https://github.com/fchollet/keras/issues/853
print("Loading Word2Vec model and generating embedding matrix...")
'''
word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    os.path.join(DATA_DIR, WORD2VEC_BIN), binary=True)
'''
with open(WORD2VEC_BIN, 'rb') as f:
    word2vec = pickle.load(f)

embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)
        




def cosine_distance(vests):
    x, y = vests
    x = K.batch_flatten(x)
    y = K.batch_flatten(y)
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1)



print("Building model...")
'''
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen,
                   weights=[embedding_weights]))
qenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True))
qenc.add(Dropout(0.3))

aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen,
                   weights=[embedding_weights]))
aenc.add(LSTM(QA_EMBED_SIZE, return_sequences=True))
aenc.add(Dropout(0.3))

# attention model
attn = Sequential()
attn.add(Merge([qenc, aenc], mode="dot", dot_axes=[1, 1]))
attn.add(Flatten())
attn.add(Dense((seq_maxlen * QA_EMBED_SIZE)))
attn.add(Reshape((seq_maxlen, QA_EMBED_SIZE)))

model = Sequential()
model.add(Merge([qenc, attn], mode="sum"))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))
'''
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen,
                   weights=[embedding_weights]))
qenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), 
                       merge_mode="sum"))
qenc.add(Dropout(0.3))
qenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, border_mode="valid"))
qenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
qenc.add(Dropout(0.3))
qenc.add(Flatten())

aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen,
                   weights=[embedding_weights]))
aenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True),
                       merge_mode="sum"))
aenc.add(Dropout(0.3))
aenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, border_mode="valid"))
aenc.add(MaxPooling1D(pool_length=2, border_mode="valid"))
aenc.add(Dropout(0.3))


unflattened_qenc = Sequential()
unflattened_qenc.add(qenc)
unflattened_qenc.add(Reshape((aenc.output_shape[1],aenc.output_shape[2])))

# attention model
attn = Sequential()
attn.add(Merge([unflattened_qenc, aenc], mode="dot", dot_axes=[1, 1]))
attn.add(Flatten())
#attn.add(Dense((seq_maxlen * QA_EMBED_SIZE)))
#attn.add(Reshape((seq_maxlen, QA_EMBED_SIZE)))
attn.add(Dense((aenc.output_shape[1]*(QA_EMBED_SIZE // 2))))
attn.add(Reshape((aenc.output_shape[1], QA_EMBED_SIZE // 2)))
attn.add(Flatten())


model = Sequential()
model.add(Merge([qenc, attn], mode="cos", dot_axes=1))


'''
model = Sequential()
model.add(Merge([qenc, attn], mode="sum"))
model.add(Flatten())
model.add(Dense(1, activation="softmax"))
'''


model.compile(optimizer="adam", loss="mean_squared_error",
              metrics=["accuracy"])

print("Training...")
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "qa-lstm-attn-best.hdf5"),
    verbose=1, save_best_only=True)
model.fit([Xqtrain, Xatrain], Ytrain, batch_size=BATCH_SIZE,
          nb_epoch=NBR_EPOCHS, validation_split=0.1,
          callbacks=[checkpoint])

print("Evaluation...")
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))

model.save_weights(os.path.join(MODEL_DIR, "qa-lstm-attn-final.hdf5"))
'''
with open(os.path.join(MODEL_DIR, "qa-lstm-attn.json"), "wb") as fjson:
    fjson.write(model.to_json())
'''
model.load_weights(filepath=os.path.join(MODEL_DIR, 
                                         "qa-lstm-attn-best.hdf5"))
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
   
