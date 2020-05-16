from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import os
import numpy as np
import tensorflow.keras.backend as K
import unicodedata
import sys

lang_ = sys.argv[1]
#text = [l.strip().split('\t') for l in open('corrected_forms.tsv','r') for i in range(10)]+[l.strip().split('\t') for l in open('training_data.tsv','r')]

fixed = [l.strip().split('\t') for l in open('corrected_forms.tsv','r') for i in range(10)]
nonfixed = [l.strip().split('\t') for l in open('training_data.tsv','r')]

corr_dict = {(l[0],l[1]):l[2] for l in fixed}
ipa_dict = {(l[0],l[1]):l[2] for l in nonfixed}

cutoff = len(fixed)

text = fixed+nonfixed

text = [l for l in text if len(l[2]) < 30]

lang_raw = [l[0] for l in text]
input_raw = [list(l[-2].replace('.','').replace('(','').replace(')','').lower()) for l in text]
output_raw = [['[']+list(l[-1].replace('.','').replace('(','').replace(')','').lower())[1:-1]+[']'] for l in text]

langs = sorted(set(lang_raw))
input_segs = sorted(set([s for w in input_raw for s in w]))
output_segs = sorted(set([s for w in output_raw for s in w]))

N = len(text)
L = len(langs)
X = len(input_segs)
Y = len(output_segs)
T_x = max([len(l) for l in input_raw])
T_y = max([len(l) for l in output_raw])

lang_id = np.zeros([N,T_x,L],dtype=np.float32)
encoder_input = np.zeros([N,T_x,X],dtype=np.float32)
decoder_input = np.zeros([N,T_y,Y],dtype=np.float32)
decoder_output = np.zeros([N,T_y,Y],dtype=np.float32)

for i in range(N):
    for j,s in enumerate(input_raw[i]):
        encoder_input[i,j,input_segs.index(s)] = 1.
        lang_id[i,j,langs.index(lang_raw[i])] = 1.
    for j,s in enumerate(output_raw[i]):
        decoder_input[i,j,output_segs.index(s)] = 1.
        if j > 0:
            decoder_output[i,j-1,output_segs.index(s)] = 1.


inds = [i for i in range(len(text)) if text[i][0] == lang_]

lang_id = lang_id[inds,:,:]
encoder_input = encoder_input[inds,:,:]
decoder_input = decoder_input[inds,:,:]
decoder_output = decoder_output[inds,:,:]

lang_id_ = Input((T_x,L))
encoder_input_ = Input((T_x,X))
decoder_input_ = Input((T_y,Y))

def nonmonotonic_alignment(args):
    h_enc,h_dec,max_encoder_seq_length,latent_dim = args
    alignment_probs = K.softmax(dot([Dense(latent_dim)(h_enc),h_dec],axes=-1,normalize=False),-2)
    return(alignment_probs)

def gen_output(args):
    alignment_probs,emission_probs = args
    return(K.sum(K.expand_dims(alignment_probs,-1)*emission_probs,-3))

def gen_emission_probs(args):
    h_enc,h_dec,max_encoder_seq_length,max_decoder_seq_length,num_decoder_tokens,hidden_dim = args
    h_enc_rep = K.tile(K.expand_dims(h_enc,-2),[1,1,max_decoder_seq_length,1])
    h_dec_rep = K.tile(K.expand_dims(h_dec,-3),[1,max_encoder_seq_length,1,1])
    h_rep = K.concatenate([h_enc_rep,h_dec_rep],-1)
    #emission probabilities
    emission_probs = Dense(num_decoder_tokens, activation='softmax')(Dense(hidden_dim*3,activation='tanh')(h_rep))
    return(emission_probs)


lang_embed = Dense(64)(lang_id_)
input_embed = Dense(64)(encoder_input_)
input_ = concatenate([lang_embed,input_embed])
h_dec = LSTM(64, return_sequences=True)(decoder_input_)
h_enc = Bidirectional(LSTM(64,return_sequences=True))(encoder_input_)
alignment_probs = Lambda(nonmonotonic_alignment)([h_enc,h_dec,T_x,64])
alignment_probs = Lambda(lambda x:x, name='attention')(alignment_probs)
emission_probs = gen_emission_probs([h_enc,h_dec,T_x,T_y,Y,64])
decoder_output_ = Lambda(gen_output)([alignment_probs,emission_probs])
model = Model([lang_id_, encoder_input_, decoder_input_], decoder_output_)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#model.save_weights('g2p.h5')
#model.load_weights('g2p_overnight.h5')
#model.load_weights('g2p_{}.h5'.format(lang_))
#for i in range(3):
    #indices = np.arange(len(text))
#    indices = np.concatenate([np.arange(cutoff),np.random.randint(cutoff,len(text),cutoff*10)])
#    model.fit([lang_id[indices], encoder_input[indices], decoder_input[indices]], decoder_output[indices], epochs=1)
#    model.fit([lang_id[:cutoff], encoder_input[:cutoff], decoder_input[:cutoff]], decoder_output[:cutoff], batch_size=32, epochs=1)
#    model.fit([lang_id, encoder_input, decoder_input], decoder_output, batch_size=32, epochs=1)
#    model.save('g2p_{}.h5'.format(lang_))


def decode_sequence(input_seq,lang_id,attn=False):
    input_seq = list(input_seq)
#    string = ['\t']
    lang_id_ = np.zeros([1,T_x,L])
    input_seq_ = np.zeros([1,T_x,X])
    for i,s in enumerate(input_seq):
        if s in input_segs:
            input_seq_[0,i,input_segs.index(s)] = 1.
        lang_id_[0,i,langs.index(lang_id)] = 1.
    lang_id = lang_id_
    input_seq = input_seq_
    string = []    
    target_seq = np.zeros((1, T_y, Y))
    target_seq[0, 0, output_segs.index('[')] = 1.
#    attention_mat
    for t in range(T_y-1):
        output_tokens = model.predict([lang_id,input_seq,target_seq])
        curr_index = np.argmax(output_tokens[:,t,:])
        target_seq[0, t+1, curr_index] = 1.
        symbol = output_segs[curr_index]
        string.append(symbol)
        if symbol == ']':
            break
    return(string)


old_IPA = [l.strip().split('\t') for l in open('old_IPA.tsv','r')]
old_IPA_key = {(l[1],l[2]):l[3] for l in old_IPA}

ocskey = """i;i
y;ɨ
u;u
ь;ɪ
ъ;ʊ
e;ɛ
o;ɔ
ę;ɛ̃
ę;ɛ̃
ǫ;ɔ̃
ǫ;ɔ̃
ě;æ
ě;æ
p;p
b;b
m;m
w;w
t;t
d;d
s;s
z;z
c;t͡s
ʒ;dz
n;n
l;l
r;r
č;t͡ʃ
š;ʃ
ž;ʒ
j;j
lʹ;ʎ
nʹ;ɲ
rʹ;rʲ
lj;ʎ
nj;ɲ
rj;rʲ
k;k
g;g
x;x"""

ocskey = ocskey.split('\n')
ocskey = [l.split(';') for l in ocskey]
ocskey = sorted(ocskey,key=lambda x:len(x[0]),reverse=True)


forms = [l.strip().split('\t') for l in open('../derksen_slavic_fixed.tsv','r')]
forms = [l for l in forms if l[3] == lang_]

for i,l in enumerate(forms):
    #if l[3] == 'russian':
        forms[i][4] = bytes(l[4],'utf8').decode('utf8').replace("'","ʹ").replace('ë','jó').replace('ĺ','lʹ')


f = open('{}_IPA.tsv'.format(lang_),'w')
decoded = []
for l in forms:
    etym = l[0]
    lang = l[3]
    orth = unicodedata.normalize('NFC',l[4]).lower()
    if lang != 'church_slavic' and lang != 'old_church_slavic':
        if (lang,orth) in corr_dict.keys():
            ll = [etym,lang,orth,corr_dict[(lang,orth)],'','CORRECTED']
        elif (lang,orth) in ipa_dict.keys():
            ll = [etym,lang,orth,ipa_dict[(lang,orth)][1:-1],'','IN_WIKTIONARY']
        else:
            decoded_form = ''.join(decode_sequence(orth,lang)[:-1])
            ll = [etym,lang,orth,decoded_form]
            if (lang,orth) in old_IPA_key.keys():
                ll.append(old_IPA_key[(lang,orth)])
                if old_IPA_key[(lang,orth)] != decoded_form:
                    ll.append('DIFFERENT')
                else:
                    ll.append('SAME')
        print('\t'.join(ll),file=f)
        print('\t'.join(ll))
    else:
        phon = orth
        for k in ocskey:
            phon = phon.replace(k[0],k[1])
        print('\t'.join([etym,lang,orth,phon]),file=f)



f.close()