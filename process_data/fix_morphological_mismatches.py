import re
import editdistance
from collections import defaultdict

text = [l.strip().split('\t') for l in open('derksen_slavic.tsv','r')]

"""Normalize reflexives: if 'se' in PSl but not in reflex or 'se' not in PSl but 'se/sja' in reflex, change PSl form"""
for i,l in enumerate(text):
    if l[1] == 'v.' and ' ' in l[0]:
        if l[3] == 'russian':
            if not l[4].endswith('sja'):
                #print(l)
                text[i][0] = l[0].split()[0] #get rid of PSl se
                #print(l)
        else:
            if ' ' not in l[4]:
                #print(l)
                text[i][0] = l[0].split()[0] #get rid of PSl se
                #print(l)
    if l[1] == 'v.' and ' ' not in l[0]:
        if l[3] == 'russian':
            if l[4].endswith('sja'):
                #print(l)
                text[i][0] = l[0] + ' sę' #add PSl se
                #print(l)
        else:
            if ' ' in l[4]:
                #print(l)
                text[i][0] = l[0] + ' sę' #get rid of PSl se
                text[i][4] = l[4].replace('(','').replace(')','') #get rid of parens
                #print(l)



"""Bulgarian verbs: for Bulgarian verbs that don't end in m, change the PSl form"""
bulg_key = [l.strip().split('\t') for l in open('blg_verb_key.tsv','r')]                
bulg_key = {l[0]:l[1] for l in bulg_key}
bulg_key.pop('děti')
                
for i,l in enumerate(text):
    if l[0]=='gǫgniti' and l[3]=='bulgarian':
        text[i][0] = 'gǫgnivъ'   


for i,l in enumerate(text):
    if l[1] == 'v.' and l[3] == 'bulgarian' and not l[4].endswith('m'):
        #"""verbs that need to be converted"""
        #print(l)
        if text[i][0] in bulg_key.keys():
            text[i][0] = bulg_key[text[i][0]]
    if l[1] == 'v.' and l[3] == 'bulgarian' and l[0] == 'jьměti':
        if text[i][0] in bulg_key.keys():
            text[i][0] = bulg_key[text[i][0]]


"""replace n-stems with accusative"""
#n_stems = {'pòlmy', 'korenь', 'remy', 'dь̑nь', 'ęčьmy', 'kamy', 'kremy'}
n_stems = {
    'pòlmy':'pòlmenь',
    'remy':'remenь',
    'ęčьmy':'ęčьmenь',
    'kamy':'kàmenь',
    'kremy':'kremenь'
}


for i,l in enumerate(text):
    if l[1] == 'm n.' and l[0] in n_stems.keys():
        #print(l)
        text[i][0] = n_stems[l[0]]


"""strong/weak adjectives: extend suffix and remove accent if reflex shows traces of strong adjective"""

accent = ['̀', '́', '̌', '̏', '̑', '̨']
def make_long(a):
    a = a+'jь'
    for b in accent:
        a=a.replace(b,'')
    return(a)


for i,l in enumerate(text):
    if l[1].startswith('adj') or l[1].startswith('num. o'):
        if l[3].endswith('sorbian'):
            text[i][0] = make_long(l[0])
        else:
            if l[3] != 'bulgarian':
                if l[4].endswith('y') or l[4].endswith('i') or l[4].endswith('j'):
                    text[i][0] = make_long(l[0])


#[l for l in text if l[1].startswith('adj') and l[-1].endswith('y') or l[1].startswith('adj') and l[-1].endswith('i') or l[1].startswith('adj') and l[-1].endswith('j')]


"""different inflectional forms: if we see the class m. o; f. ā, etc., change to a ъ-final form, then remove redundant lines later"""

##[l[0] for l in text if l[1]=='m. o; f. ā' or l[1]=='f. ā; m. o' or ]
##get rid of ěždžь
#text = [l for l in text if l[0]!='ěždžь']
#
#variant = {}
##for each pair of etyma, 
#for x in set([l[0] for l in text]):
#  for y in set([l[0] for l in text]):
#    if x != y: 
#      if x[:-2]+'ь'==y or x[:-1]+'ь'==y:
#        variant[x]=y
#      if x[:-2]+'ъ'==y or x[:-1]+'ъ'==y:
#        variant[x]=y

##not sure what to do here

variant_forms = {}
problem_forms = {}
for e,g,v in set([tuple([l[0],l[1],l[6]]) for l in text if ';' in l[1]]):
    genders = g.split('; ')
    variants = v.split('; ')
    if len(genders) == len(variants) and len(set(genders)) == len(genders):
        variant_forms[e] = {}
        for i,w in enumerate(genders):
            #variant_forms[e][w.replace('.','')] = variants[i].replace('I','').strip()
            variant_forms[e][w.replace('.','').split()[1]] = variants[i].replace('I','').strip()
    else:
        problem_forms[e] = {}
        problem_forms[e] = (tuple([w.replace('.','').split()[1] for w in genders[:min(len(genders),len(variants))] if '?' not in w]),tuple([w.replace('I','').strip() for w in variants[:min(len(genders),len(variants))]]))
        


#variant_forms['černь'] = {'m o':'černъ','m jo':'černь','f ā':'černa'}
#variant_forms['krina'] = {'f ā':'krina','f jā':'krinica','Accsf jā':'krinica'}
#variant_forms['krinica'] = {'f ā':'krina','f jā':'krinica','Accsf jā':'krinica'}
variant_forms['černь'] = {'o':'černъ','jo':'černь','ā':'černa'}
variant_forms['krina'] = {'ā':'krina','jā':'krinica'}
variant_forms['krinica'] = {'ā':'krina','jā':'krinica'}

problem_forms['kъkъn̨ь'] = (('jā',), ('kъk(ъ)n̨ь',))
problem_forms['kъkn̨ь'] = (('jā',), ('kъk(ъ)n̨ь',))
problem_forms.pop('černь')
problem_forms.pop('krina')
problem_forms.pop('krinica')


for i,l in enumerate(text):
    if l[0] in variant_forms.keys():
        if ' ' in l[5]:
            g,cl = l[5][1:-1].split()
            g = g[-1]
            #pos = ' '.join([g,cl])
        else:
            g = l[5][1:-1]
        if cl in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]][cl]
        elif cl == 'jā' and 'ā' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['ā']
        elif cl == 'ā' and 'jā' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['jā']
        elif cl == 'jo?' and 'io' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['io']
        elif cl == 'jo' and 'o' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['o']
        elif cl == 'o' and 'jo' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['jo']
        elif cl == 'o' and 'io' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['io']
        elif cl == 'iā' and 'jā' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['jā']
        elif cl == '(j)o' and 'o' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['jo']
        elif cl=='o/u' and 'o' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['o']
        elif cl == 'jā' and 'iā' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['iā']
        elif cl == 'n' and 'o' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['o']
        elif cl == 'i/jā' and 'jā' in variant_forms[l[0]].keys():
            text[i][0] = variant_forms[l[0]]['jā']
        elif cl == 'i/jā' and 'jā' not in variant_forms[l[0]].keys():
            if l[0][-1] != '̀':
                text[i][0] = l[0][:-1]+'ь'
            else:
                text[i][0] = l[0][:-2]+'ь'
        elif cl == 'i':
            if l[0][-1] != '̀':
                text[i][0] = l[0][:-1]+'ь'
            else:
                text[i][0] = l[0][:-2]+'ь'
        elif cl == 'm':
            if l[0][-1] != '̀':
                text[i][0] = l[0][:-1]+'ъ'
            else:
                text[i][0] = l[0][:-2]+'ъ'
        #a few stray forms to fix  ъ






#strip I.
#černь : černъ, černь
#krina : krina, krinica
#krinica : krina, krinica


for i,l in enumerate(text):
    if l[0] in problem_forms.keys():
        #print(l)
        if len(problem_forms[l[0]][0]) == 1:
            text[i][0] = problem_forms[l[0]][1][0]
        else:
            forms = [problem_forms[l[0]][1][i] for i in range(len(problem_forms[l[0]][1])) if problem_forms[l[0]][0][i] == l[5][1:-1].split()[1].strip('?')]
            if len(forms) == 1:
                text[i][0] = forms[0]
            else:
                if len(forms)==0:
                    forms = [problem_forms[l[0]][1][i] for i in range(len(problem_forms[l[0]][1])) if problem_forms[l[0]][0][i] in l[5][1:-1].split()[1].strip('?')]
                text[i][0] = sorted(forms,key=lambda x:editdistance.distance(x,l[4]))[0]
        #print(l)

        

"""prefixes"""
#sorted(set([(l[0][0],l[4][0]) for l in text if l[0][0] != l[4][0]]))

for i,l in enumerate(text):
    if l[4] == "zadzierzgnąć":
        text[i][0] = 'za' + l[0] #??
    if l[4] == "otvérgnut'":
        text[i][0] = 'ot' + l[0]
    if l[4] == "očrěsti":
        text[i][0] = 'ob' + l[0]


text = [l for l in text if l[4] != "Dɫugosiodɫo"]

#"zadzierzgnąć"
#"jadžgár"
#"otvérgnut'"
#"Dɫugosiodɫo"

"""deal with ocs and cs prefixes"""

#"namaiaaxǫ"

#'ozoba'
#'očrěsti' -> PSl 'občersti'

for i,l in enumerate(text):
    if l[4].startswith('iz') or l[4].startswith('is') and not l[0].startswith('iz') and not l[0].startswith('is'):
        text[i][0] = 'jьz' + l[0]
    if l[3].endswith('church_slavic'):
        if l[4].startswith('u'):
            if not l[0].startswith('ju') and not l[0].startswith('ū') and not l[0].startswith('u'):
                text[i][0] = 'u'+l[0]
        if l[4].startswith('po') and not l[0].startswith('p'):
            text[i][0] = 'po' + l[0]
        if l[4].startswith('pa') and not l[0].startswith('p'):
            text[i][0] = 'pa' + l[0]
        if l[4].startswith('pro') and not l[0].startswith('p'):
            text[i][0] = 'pro' + l[0]
        if l[4].startswith('pra') and not l[0].startswith('p'):
            text[i][0] = 'pra' + l[0]
        if l[4].startswith('vъs') and not l[0].startswith('vъs'):
            text[i][0] = 'vъz' + l[0] #???? NOT SURE


"""parentheses: get rid of information in parentheses in reflexes"""
for i,l in enumerate(text):
    text[i][4] = re.sub('\(.+\)','',text[i][4])


"""get rid of m-final Bulgarian verbs"""
        
for i in range(len(text))[::-1]:
    l = text[i]
    if l[1] == 'v.' and l[3] == 'bulgarian' and l[4].endswith('m'):
        text.pop(i)
        

"""miscellanea"""


for i,l in enumerate(text):
    if l[3] == 'old_church_slavic' and l[4] == 'žrьcь':
        text[i][4] = 'žьrьcь'
    if l[3] == 'old_church_slavic' and l[4] == 'prьsi':
        text[i][0] = 'pь̀rsь' # replaces PSl Nsgf by Nplf as it is given for OCS; note that this etymon is also affected by ocs_syll_liqu_rplc.py
    if l[3] == 'bcs' and l[4] == 'pȑsi':
        text[i][0] = 'pь̀rsь' # replaces PSl Nsgf by Nplf as it is given for BCS; note that this etymon is also affected by ocs_syll_liqu_rplc.py
    if l[3] == 'russian' and l[4] == 'pérsi':
        text[i][0] = 'pь̀rsь' # replaces PSl Nsgf by Nplf as it is given for Russian; note that this etymon is also affected by ocs_syll_liqu_rplc.py
    if l[3] == 'slovene' and l[4] == 'pŕsi':
        text[i][0] = 'pь̀rsь' # replaces PSl Nsgf by Nplf as it is given for Slovene; note that this etymon is also affected by ocs_syll_liqu_rplc.py
    #if l[3] == 'old_church_slavic' and l[4] == 'vrьtitъ sę':
    #    text[i][0] = 'vr̩těti sę' # replaces PSl Nsgf by Nplf as it is given for OCS; note that this etymon is also affected by ocs_syll_liqu_rplc.py



for i,l in enumerate(text):
    if l[3] == 'old_church_slavic':
        if l[4] == 'črъnъ':
            text[i][4] = 'čr̩nъ'
        if l[4] == 'črъnъ':
            text[i][4] = 'čr̩nъ'
        if l[4] == 'črъmьnъ':
            text[i][4] ='čr̩mьnъ'
        if l[4] == 'črъta':
            text[i][4] = 'čr̩ta'
        if l[4] == 'črъvь':
            text[i][4] = 'čr̩vь'
        if l[4] == 'grъdъ':
            text[i][4] = 'gr̩dъ'
        if l[4] == 'grъnilъ':
            text[i][4] = 'gr̩nilъ'
        if l[4] == 'grъstijо̨':
            text[i][4] = 'gr̩stijо̨'
        if l[4] == 'krъma':
            text[i][4] = 'kr̩ma'
        if l[4] == 'mrъzěti':
            text[i][4] = 'mr̩zěti'
        if l[4] == 'oskrъdъ':
            text[i][4] = 'oskr̩dъ'
        if l[4] == 'isprъgnǫti':
            text[i][4] = 'ispr̩gnǫti'
        if l[4] == 'žrъny':
            text[i][4] = 'žr̩ny'
        if l[4] == 'brьzo':
            text[i][4] = 'br̩zo'
        if l[4] == 'drьzъ':
            text[i][4] = 'dr̩zъ'
        if l[4] == 'drьzati':
            text[i][4] = 'dr̩zati'
        if l[4] == 'drьznǫti':
            text[i][4] = 'dr̩znǫti'
        if l[4] == 'drьžati':
            text[i][4] = 'dr̩žati'
        if l[4] == 'mrьtvъ':
            text[i][4] = 'mr̩tvъ'
        if l[4] == 'mrьknǫti':
            text[i][4] = 'mr̩knǫti'
        if l[4] == 'prьsi':
            text[i][4] = 'pr̩si'
        if l[4] == 'prьstъ':
            text[i][4] = 'pr̩stъ'
        if l[4] == 'prьstь':
            text[i][4] = 'pr̩stь'
        if l[4] == 'prьvъ':
            text[i][4] = 'pr̩vъ'
        if l[4] == 'srьdьce':
            text[i][4] = 'sr̩dьce'
        if l[4] == 'sъmrьtь':
            text[i][4] = 'sъmr̩tь'
        if l[4] == 'smrьděti':
            text[i][4] = 'smr̩děti'
        if l[4] == 'tvrьdъ':
            text[i][4] = 'tvr̩dъ'
        if l[4] == 'tvrьdь':
            text[i][4] = 'tvr̩dь'
        if l[4] == 'vrьxъ':
            text[i][4] = 'vr̩xъ'
        if l[4] == 'vrьsta':
            text[i][4] = 'vr̩sta'
        if l[4] == 'vrьtitъ sę':
            text[i][4] = 'vr̩titъ sę'
        if l[4] == 'zrьno':
            text[i][4] = 'zr̩no'
        if l[4] == 'žrьti':
            text[i][4] = 'žr̩ti'
        if l[4] == 'dlьgъ':
            text[i][4] = 'dl̩gъ'
        if l[4] == 'mlьčati':
            text[i][4] = 'ml̩čati'
        if l[4] == 'plьnъ':
            text[i][4] = 'pl̩nъ'
        if l[4] == 'plьzati':
            text[i][4] = 'pl̩zati'
        if l[4] == 'vlьkъ':
            text[i][4] = 'vl̩kъ'
        if l[4] == 'vlьna':
            text[i][4] = 'vl̩na'
        if l[4] == 'dlъgota':
            text[i][4] = 'dl̩gota'
        if l[4] == 'mlъni':
            text[i][4] = 'ml̩ni'
        if l[4] == 'mlъva':
            text[i][4] = 'ml̩va'
        if l[4] == 'mlъviti':
            text[i][4] = 'ml̩viti'
        if l[4] == 'plъkъ':
            text[i][4] = 'pl̩kъ'
        if l[4] == 'slъnьce':
            text[i][4] = 'sl̩nьce'
        if l[4] == 'zlъčь':
            text[i][4] = 'zl̩čь'


for i,l in enumerate(text):
	if l[3] == 'upper_sorbian':
		if l[4] == 'jězer': 
			text[i][4] = 'jězor'
		if l[4] == 'gněw':
			text[i][4] = 'hněw'
		if l[4] == 'ɫoni': 
			text[i][4] = 'loni'
		if l[4] == 'hójić': 
			text[i][4] = 'hojić'
		if l[4] == 'rośc': 
			text[i][4] = 'rosć'



text = [l for l in text if len(l) == 7]
for i,l in enumerate(text):
    if l[4] == 'Vaccinium myrtillus':
        if l[3] == 'bcs':
            text[i][4] = 'brùsnica'
    if l[4] == 'Vaccinium vitis-idaea':
        if l[3] == 'bulgarian':
            text[i][4] = 'brusníca'
        if l[3] == 'czech':
            text[i][4] = 'brusnice'
        if l[3] == 'polish':
            text[i][4] = 'brusznica'
        if l[3] == 'russian':
            text[i][4] = 'brusníka'
        if l[3] == 'slovak':
            text[i][4] = 'brusnica'
        if l[3] == 'slovene':
            text[i][4] = 'brusníca'
    if l[4] == 'D.':
        if l[3] == 'russian':
            text[i][4] = 'déva'
    if l[4] == 'Genista':
        if l[3] == 'russian':
            text[i][4] = 'drok'
    if l[4] == 'Hordeum distichum':
        if l[3] == 'bcs':
            text[i][4] = 'glȍta'
    if l[4] == 'Brachypodium':
        if l[3] == 'slovene':
            text[i][4] = 'glǫ̑ta'
    if l[4] == 'Ulmus montana':
        if l[3] == 'russian':
            text[i][4] = "íl'm"
    if l[4] == 'Ulmus montana':
        if l[3] == 'ukrainian':
            text[i][4] = "il'm"
            

for i,l in enumerate(text):
    text[i][4] = text[i][4].lower()


unique = defaultdict(list)
for l in text:
    unique[(l[0],l[3])].append(tuple(l))


new_text = []
for k in unique.keys():
    if len(set(unique[k]))==1:
        new_text.append(unique[k][0])
    else:
        forms = sorted(list(set(unique[k])),key=lambda x:editdistance.distance(k[0],x[4]))[0]
        new_text.append(forms)


final_text = []
unique = defaultdict(list)
for l in new_text:
    if ';' in l[-1]:
        unique[(l[-1],l[3])].append(tuple(l))
    else:
        final_text.append(tuple(l))


for k in unique.keys():
    if len(set(unique[k]))==1:
        final_text.append(unique[k][0])
    else:
        forms = sorted(list(set(unique[k])),key=lambda x:editdistance.distance(x[0],x[4]))[0]
        final_text.append(forms)
        

#sorted(forms,key=lambda x:editdistance.distance(x,l[4]))[0]
#text = sorted(set([tuple(l) for l in text]))


f = open('derksen_slavic_fixed.tsv','w')
for l in final_text:
    print('\t'.join(l),file=f)


f.close()