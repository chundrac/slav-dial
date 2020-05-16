import sys
import random
import numpy as np
from util import *
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Lambda, dot, RepeatVector, Concatenate
import tensorflow as tf
import tensorflow.keras.backend as K
import larq as lq
import editdistance

assert(len(sys.argv)==2), "usage: python3 run_model batch_to_hold_out embedding_type"

lang = sys.argv[1]
batch = 'all'

mode = 'ST'

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
dec_output_ = monotonic_alignment([h_enc,h_dec,T_x,T_y,Y,hidden_dim])
dec_output_ = Lambda(lambda x:x)(dec_output_)

decoder = Model([latent_inputs_,enc_input_,dec_input_],dec_output_)
output = decoder([prior(lang_id_),enc_input_,dec_input_])
model = Model(inputs=[lang_id_,enc_input_,dec_input_], outputs=output)
#model.compile(optimizer='adam',loss='categorical_crossentropy')
#model.fit([lang_id,enc_in,dec_in],dec_out,batch_size=256,epochs=200)
#model.save_weights('model_{}_weights_{}.h5'.format(mode,batch))
model.load_weights('model_{}_weights_{}.h5'.format(mode,batch))


def decode_sequence2(input_seq,vector,attn=False):
    input_seq = list(input_seq)
    input_seq_ = np.zeros([1,T_x,X])
    for i,s in enumerate(input_seq):
        if s in input_segs:
            input_seq_[0,i,input_segs.index(s)] = 1.
    input_seq = input_seq_
    string = ['[']    
    target_seq = np.zeros((1, T_y, Y))
    target_seq[0, 0, output_segs.index('[')] = 1.
    #print(vector.shape,input_seq.shape)
#    attention_mat
    for t in range(T_y-1):
        output_tokens = decoder.predict([vector,input_seq,target_seq])
        curr_index = np.argmax(output_tokens[:,t,:])
        target_seq[0, t+1, curr_index] = 1.
        symbol = output_segs[curr_index]
        string.append(symbol)
        if symbol == ']':
            break
    return(string)


Z = prior.predict(np.eye(L))


def perturb(lang,str):
    vector = Z[langs.index(lang):langs.index(lang)+1]
    pred_reflex = (''.join(decode_sequence2(str,vector)))
    neighbors = []
    for i in range(128):
      vec = vector
      vec[0,i] = abs(vec[0,i]-1)
      neighbors.append(''.join(decode_sequence2(str,vec)))
    return(pred_reflex,sorted(set(neighbors)))


inds = [i for i in list(range(N)) if lang_raw['train'][i] == lang]
inds = random.sample(inds,100)


for i in inds:
    inp = ''.join(input_raw['train'][i])
    out = ''.join(output_raw['train'][i])
    pred_reflex,neighbors = perturb(lang,''.join(input_raw['train'][i]))
    line = [lang,inp,out,pred_reflex]+neighbors
    f = open('ST_{}_nearest_neighbors.tsv'.format(lang),'a')
    print('\t'.join(line),file=f)
    f.close()
    