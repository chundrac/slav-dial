import sys
import random
import numpy as np
from util import *
from collections import defaultdict

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Lambda, dot, RepeatVector, Concatenate
import tensorflow as tf
import tensorflow.keras.backend as K
import larq as lq
import editdistance

assert(len(sys.argv)==3), "usage: python3 run_model batch_to_hold_out embedding_type"

batch = sys.argv[1]
if batch != 'all':
    batch = int(batch)

mode = sys.argv[2]

#mode = 'ST'
batch = 'all'

lang_raw,input_raw,output_raw,langs,input_segs,output_segs,X,Y,T_x,T_y,L,N,lang_id,enc_in,dec_in,dec_out = generate_data(batch)
print(L)

latent_dim = 128
embed_dim = 128
hidden_dim = 128
batch_size = 256

lang_id_ = Input((L,))
enc_input_ = Input(shape=(T_x,X))
dec_input_ = Input(shape=(T_y,Y))
enc_mask = tf.expand_dims(tf.reduce_sum(enc_input_,-1),-1)
dec_mask = tf.expand_dims(tf.reduce_sum(dec_input_,-1),-1)

if mode == 'ST':
    z_ = Dense(latent_dim,activation='tanh',use_bias=False)(lang_id_)
    z = lq.quantizers.SteSign()(z_)*.5+.5

if mode == 'dense':
    z = Dense(latent_dim,use_bias=False)(lang_id_)

if mode == 'sigmoid':
    z = Dense(latent_dim,activation='sigmoid',use_bias=False)(lang_id_)

prior = Model(inputs = lang_id_,outputs = z)

latent_inputs_ = Input((latent_dim,))
rep_latent_inputs_ = RepeatVector(T_x)(latent_inputs_)
embed_inputs_ = Concatenate()([enc_input_,rep_latent_inputs_])
embedding = Dense(embed_dim)(embed_inputs_)
h_enc = Bidirectional(LSTM(hidden_dim, return_sequences=True, activation=None),'concat')(embedding)*enc_mask
h_dec = LSTM(hidden_dim, return_sequences=True, activation=None)(dec_input_)*dec_mask
#alignment_probs_,emission_probs = monotonic_alignment([h_enc,h_dec,T_x,T_y,Y,hidden_dim])
struc_zeros = K.expand_dims(K.cast(np.triu(np.ones([T_x,T_x])),dtype='float32'),0)
alignment_probs = K.softmax(dot([Dense(hidden_dim)(h_enc),h_dec],axes=-1,normalize=False),-2)
h_enc_rep = K.tile(K.expand_dims(h_enc,-2),[1,1,T_y,1])
h_dec_rep = K.tile(K.expand_dims(h_dec,-3),[1,T_x,1,1])
h_rep = K.concatenate([h_enc_rep,h_dec_rep],-1)
alignment_probs_ = []
for i in range(T_y):
    if i == 0:
        align_prev_curr = tf.gather(alignment_probs,i,axis=-1)
    if i > 0:
        align_prev_curr = tf.einsum('nx,ny->nxy',tf.gather(alignment_probs,i,axis=-1),alignment_probs_[i-1])
        align_prev_curr *= struc_zeros
        align_prev_curr = K.sum(align_prev_curr,1)+1e-6
        align_prev_curr /= K.sum(align_prev_curr,-1,keepdims=True)
    alignment_probs_.append(align_prev_curr)


alignment_probs_ = K.stack(alignment_probs_,-1)
emission_probs = Dense(hidden_dim*3,activation='tanh')(h_rep)
emission_probs = Dense(Y, activation='softmax')(emission_probs)

alignment_probs_ = Lambda(lambda x:x)(alignment_probs_)
alignment_model = Model([latent_inputs_,enc_input_,dec_input_],alignment_probs_)
alignment_output = alignment_model([prior(lang_id_),enc_input_,dec_input_])
alignment = Model([lang_id_,enc_input_,dec_input_],alignment_output)

alphas = tf.expand_dims(alignment_probs_,-1)*emission_probs
dec_output_ = tf.reduce_sum(alphas,-3)
dec_output_ = Lambda(lambda x:x)(dec_output_)

decoder = Model([latent_inputs_,enc_input_,dec_input_],dec_output_)
output = decoder([prior(lang_id_),enc_input_,dec_input_])
model = Model(inputs=[lang_id_,enc_input_,dec_input_], outputs=output)
#model.compile(optimizer='adam',loss='categorical_crossentropy')
#model.fit([lang_id,enc_in,dec_in],dec_out,batch_size=256,epochs=20)
#model.save_weights('model_{}_weights_{}.h5'.format(mode,batch))
model.load_weights('model_{}_weights_{}.h5'.format(mode,batch))


def get_aligned_pairs(lang_id,input_seq,output_seq):
    lang_id_ = np.zeros([1,L])
    input_seq_ = np.zeros([1,T_x,X])
    lang_id_[0,langs.index(lang_id)] = 1.
    for i,s in enumerate(input_seq):
        if s in input_segs:
            input_seq_[0,i,input_segs.index(s)] = 1.
    output_seq_ = np.zeros([1,T_y,Y])
    for i,s in enumerate(output_seq):
        if s in output_segs:
            output_seq_[0,i,output_segs.index(s)] = 1.
    #lang_id = lang_id_
    #input_seq = input_seq_
    #output_seq = output_seq_
    att = alignment([lang_id_,input_seq_,output_seq_])
    att = att[:,:len(input_seq),:len(output_seq)-1]
    aligned = np.zeros_like(att)
    aligned[np.where(att==np.max(att,1))]=1
    aligned=aligned[0]
    pairs = []
    for i in range(aligned.shape[0]):
        edit = []
        for j in aligned[i].nonzero()[0]:
            edit.append(output_seq[j+1])
        pairs.append((input_seq[i],tuple(edit)))
    return(pairs)


text = []
for i in range(10):
  for l in open('decoded_{}_{}.tsv'.format(i),'r'):
    text.append(l.strip().split('\t'))



correct = defaultdict(list)
wrong = defaultdict(list)
for l in text:
    if text.index(l) in range(0,N,1000):
        print(text.index(l))
    pairs = get_aligned_pairs(l[0],list(l[1]),list(l[2]))
    for p in pairs:
        correct[l[0]].append(p)
    if float(l[4]) > 0:
        pairs_wrong = get_aligned_pairs(l[0],list(l[1]),list(l[3]))
        for i in range(len(pairs_wrong)):
            if pairs_wrong[i] != pairs[i]:
                wrong[l[0]].append(pairs_wrong[i])


error_types = defaultdict(int)
for k in wrong.keys():
    for e in wrong[k]:
        if e in correct[k]:
            error_types['same_lang'] += 1
        elif e in [v for l in wrong.keys() for v in wrong[l] if l != k]:
            error_types['other_lang'] += 1
        else:
            error_types['silly'] += 1


for k in error_types.keys():
    print(k,error_types[k]/sum(error_types.values()))