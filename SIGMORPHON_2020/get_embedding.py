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

assert(len(sys.argv)==2), "usage: python3 run_model batch_to_hold_out embedding_type"

batch = 'all'

#batch = sys.argv[1]
#if batch != 'all':
#    batch = int(batch)

mode = sys.argv[1]

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

Z = prior(np.eye(L))

f = open('tree_materials/{}_embeddings.txt'.format(mode),'w')
for i in range(L):
    print(' '.join([langs[i]]+[str(j) for j in list(np.array(Z[i,:]))]),file=f)


f.close()