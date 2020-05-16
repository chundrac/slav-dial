import numpy as np
import random
from collections import defaultdict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Lambda, dot, RepeatVector, Concatenate
import tensorflow as tf
import tensorflow.keras.backend as K

def generate_data(k):
    if k != 'all':
        k = int(k)
    K = 10
    random.seed(1234)
    #load data
    text = []
    for l in open('slavic_all_IPA.tsv','r'):
        text.append(l.strip('\n').split('\t'))
    text = [l for l in text if l[1] != 'macedonian']
    #group by language
    lang_list = defaultdict(list)
    for l in text:
        lang_list[l[1]].append(l)
    [random.shuffle(lang_list[k]) for k in lang_list.keys()]
    lang_batches = defaultdict(list)
    for key in lang_list.keys():
        breaks = list(range(0,len(lang_list[key]),int(len(lang_list[key])/K)+1))+[len(lang_list[key])]
        for i,b in enumerate(breaks[:-1]):
            lang_batches[key].append(lang_list[key][breaks[i]:breaks[i+1]])        
    train = [l for v in lang_batches.values() for i in range(K) for l in v[i] if i != k]
    test = [l for v in lang_batches.values() for i in range(K) for l in v[i] if i == k]
    langs = list(lang_list.keys())
    input_segs = sorted(set([s for l in text for s in list(l[0])]))
    output_segs = sorted(set([s for l in text for s in ['[']+list(l[3])+[']']]))
    lang_raw = defaultdict(list)
    input_raw = defaultdict(list)
    output_raw = defaultdict(list)
    lang_raw['train'] = [l[1] for l in train]
    lang_raw['test'] = [l[1] for l in test]
    input_raw['train'] = [list(l[0]) for l in train]
    input_raw['test'] = [list(l[0]) for l in test]
    output_raw['train'] = [['[']+list(l[3])+[']'] for l in train]
    output_raw['test'] = [['[']+list(l[3])+[']'] for l in test]
    X = len(input_segs)
    Y = len(output_segs)
    T_x = max([len(l) for l in input_raw['train']+input_raw['test']])
    T_y = max([len(l) for l in output_raw['train']+output_raw['test']])
    L = len(langs)
    N = len(train)
    lang_id = np.zeros([N,L],dtype='float32')
    for i,l in enumerate(lang_raw['train']):
        lang_id[i,langs.index(l)] = 1.
    enc_in  = np.zeros([N,T_x,X],dtype='float32')
    for i,l in enumerate(input_raw['train']):
        for j,s in enumerate(l):
            enc_in[i,j,input_segs.index(s)] = 1.
    dec_in  = np.zeros([N,T_y,Y],dtype='float32')
    dec_out = np.zeros([N,T_y,Y],dtype='float32')
    for i,l in enumerate(output_raw['train']):
        for j,s in enumerate(l):
            dec_in[i,j,output_segs.index(s)] = 1.
            if j > 0:
                dec_out[i,j-1,output_segs.index(s)] = 1.
    return(lang_raw,input_raw,output_raw,langs,input_segs,output_segs,X,Y,T_x,T_y,L,N,lang_id,enc_in,dec_in,dec_out)


def monotonic_alignment(args):
    h_enc,h_dec,T_x,T_y,Y,hidden_dim = args
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
    #alphas = tf.expand_dims(alignment_probs_,-1)*emission_probs
    #return(tf.reduce_sum(alphas,-3))
    return(alignment_probs_,emission_probs)


def decode_sequence(input_seq,lang_id,langs,output_segs,input_segs,L,X,Y,T_x,T_y,model,temp=False):
    lang_id_ = np.zeros([1,L])
    input_seq_ = np.zeros([1,T_x,X])
    lang_id_[0,langs.index(lang_id)] = 1.
    for i,s in enumerate(input_seq):
        if s in input_segs:
            input_seq_[0,i,input_segs.index(s)] = 1.
    lang_id = lang_id_
    input_seq = input_seq_
    string = ['[']    
    target_seq = np.zeros((1, T_y, Y))
    target_seq[0, 0, output_segs.index('[')] = 1.
#    attention_mat
    for t in range(T_y-1):
        if temp == False:
            output_tokens = model.predict([lang_id,input_seq,target_seq])
        else:
            output_tokens = model.predict([lang_id,input_seq,target_seq,np.ones(1,)*temp])
        curr_index = np.argmax(output_tokens[:,t,:])
        target_seq[0, t+1, curr_index] = 1.
        symbol = output_segs[curr_index]
        string.append(symbol)
        if symbol == ']':
            break
    return(string)
