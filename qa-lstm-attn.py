
# See https://stackoverflow.com/questions/49941903/keras-compute-cosine-distance-between-two-flattened-outputs

# -*- coding: utf-8 -*-
from __future__ import division, print_function
#import gensim
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout, Reshape, Flatten, Lambda, Dot
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from sklearn.cross_validation import train_test_split
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras import backend as K
from keras.objectives import cosine_proximity
from keras import optimizers

import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import kaggle

import pickle


DATA_DIR = "data/comp_data"
MODEL_DIR = "data/models"
WORD2VEC_BIN = "/home/hcl/Documents/work/keyword-algorithms/process_reddit/model2/reddit_w2v_normalized.pickle"
WORD2VEC_EMBED_SIZE = 300

QA_TRAIN_FILE = "train.json"#"studystack_qa_cleaner_no_qm.txt"
QA_TRAIN_FILE2 = "test.json"
QA_TRAIN_FILE3 = "valid.json"


QA_EMBED_SIZE = 64
BATCH_SIZE = 32
NBR_EPOCHS = 50

## extract data

print("Loading and formatting data...")
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
        



print("Building model...")

#question
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen))
qenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), 
                       merge_mode="sum"))
#qenc.add(Dropout(0.15))
qenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, padding="valid",activation='relu'))
qenc.add(MaxPooling1D(pool_size=2, padding="valid"))
#qenc.add(Dropout(0.15))

# answer
aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,
                   input_length=seq_maxlen))
aenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True),
                       merge_mode="sum"))
#aenc.add(Dropout(0.15))
aenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, padding="valid",activation='relu'))
aenc.add(MaxPooling1D(pool_size=2, padding="valid"))
#aenc.add(Dropout(0.15))


# attention model

#notice that I'm taking "tensors" qenc.output and aenc.output
#I'm not passing "models" to a layer, I'm passing tensors 

attOut = Dot(axes=1)([qenc.output, aenc.output]) 
    #shape = (samples,QA_EMBED_SIZE//2, QA_EMBED_SIZE//2)
    #I really don't understand this output shape.... 
    #I'd swear it should be (samples, 1, QA_EMBED_SIZE//2)
attOut = Flatten()(attOut) #shape is now only (samples,)
attOut = Dense((qenc.output_shape[1]*(QA_EMBED_SIZE // 2)),activation='tanh')(attOut)
attOut = Reshape((qenc.output_shape[1], QA_EMBED_SIZE // 2))(attOut) 


#    Notice the output shape: (samples, (seq_maxlen-4)/2, QA_EMBED_SIZE // 2).
#    Notice also that this attention part requires two inputs

#Now, you can flatten the outputs of qenc and attn, no problem, you just can't do it "inside" the qenc model.

flatAttOut = Flatten()(attOut)
flatQencOut = Flatten()(qenc.output)
similarity = Dot(axes=1,normalize=True)([flatQencOut,flatAttOut])

model = Model([qenc.input,aenc.input],similarity)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def balanceLoss(yTrue,yPred):

    loss = mean_squared_error(yTrue,yPred)
    scaledTrue = (3*yTrue) + 1 
        #true values are 4 times worth the false values
        #contains 4 for true and 1 for false

    return scaledTrue * loss


model.compile(optimizer="adam", loss=balanceLoss,
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
   
