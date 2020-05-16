import sys
import random
import numpy as np
from util import *

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
model.compile(optimizer='adam',loss='categorical_crossentropy')
model.fit([lang_id,enc_in,dec_in],dec_out,batch_size=256,epochs=200)
model.save_weights('model_{}_weights_{}.h5'.format(mode,batch))
#model.load_weights('model_{}_weights_{}.h5'.format(mode,batch))

for l,i,o in zip(lang_raw['test'],input_raw['test'],output_raw['test']):
    f = open('decoded_{}_{}.tsv'.format(mode,batch),'a')
    o_ = decode_sequence(i,l,langs,output_segs,input_segs,L,X,Y,T_x,T_y,model)
    d = editdistance.distance(o,o_)/max([len(o),len(o_)])
    print('\t'.join([l,''.join(i),''.join(o),''.join(o_),str(d)]),file=f)
    f.close()