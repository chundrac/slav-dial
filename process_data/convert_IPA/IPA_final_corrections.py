import unicodedata

text = [l.strip().split('\t') for l in open('slavic_IPA_corrected_FINAL.tsv')]

forms = []

for l in text:
    if len(l)==4:
        forms.append(l)
    if len(l)==6:
        forms.append(l[:4])
    if len(l)==7:
        if l[-1] == '1':
            forms.append(l[:4])
        elif l[-1] == '2':
            forms.append(l[:3]+[l[4]])
        else:
            forms.append(l[:3]+[l[-1]])
    if len(l)==8:
        forms.append(l[:3]+[l[-1]])
        
        
        

#segs = sorted(set([s for l in forms for s in l[3]]))

for i,l in enumerate(forms):
    forms[i][3] = unicodedata.normalize('NFC',l[3])

#fix glides:
for i,l in enumerate(forms):
    forms[i][3] = l[3].replace('u̯','w')
    forms[i][3] = l[3].replace('ʊ̯','w')
    forms[i][3] = l[3].replace('i̯','j')
    forms[i][3] = l[3].replace('ɪ̯','j')
    forms[i][3] = l[3].replace('ǝ','ə')
    forms[i][3] = l[3].replace('g','ɡ')
    forms[i][3] = l[3].replace(':','ː')
    forms[i][3] = l[3].replace('(','')
    forms[i][3] = l[3].replace(')','')
    forms[i][3] = l[3].replace('⁽','')
    forms[i][3] = l[3].replace('⁾','')
    forms[i][3] = l[3].replace('-','')
    forms[i][3] = l[3].replace('̪','')
    forms[i][0] = l[0].replace('-','')
    forms[i][0] = l[0].replace('?','')
    forms[i][0] = l[0].replace('(','')
    forms[i][0] = l[0].replace(')','')
    forms[i][0] = l[0].replace("'",'̨')
    forms[i][0] = l[0].replace('̹','̨')


segs = sorted(set([s for l in forms for s in l[3]]))


#for i,l in enumerate(forms):
#    w = list(l[3])
#    for j,s in enumerate(w):
#        if unicodedata.name(s).startswith('COMBINING') and :
#            w[j] = ''
#            #w[j-1] = unicodedata.lookup(unicodedata.name(w[j-1])+' WITH '+' '.join(unicodedata.name(s).split()[1:]))
#            w[j-1] = unicodedata.lookup(unicodedata.name(w[j-1])+' WITH '+unicodedata.name(s).split()[1])


#for i,l in enumerate(forms):
#    if '̯' in l[3]:
#        print(l[3])
#        print (l[3][l[3].index('̯')-1])


#tokens = [ɐ̯]


#add primary stress mark if not present
for i,l in enumerate(forms):
    if l[1] not in ['bcs','church_slavic','old_church_slavic','slovene']:
        if 'ˈ' not in l[3]:
            forms[i][3] = 'ˈ'+l[3]



#change slovene accent


#The acute equals a long rising tone and the circumflex a long falling tone. Gravis equals a short falling tone. But I am not sure whether the mix of wictionary- and Pleteršnik-data creates problems.
for i,l in enumerate(forms):
    if l[1] == 'slovene':
        #print (forms[i])
        x = ''
        x = ''
        if len([s for s in list(l[2]) if 'ACUTE' in unicodedata.name(s)]) > 0:
            if len([t for t in l[3] if 'GRAVE' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'GRAVE' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'GRAVE' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('GRAVE','CARON'))
                else:
                    x = '̀'
                    y = '̌'
                if 'ː' not in l[3]:
                    y += 'ː'
                forms[i][3] = l[3].replace(x,y)
            elif len([t for t in l[3] if 'CARON' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'CARON' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'CARON' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('CARON','CARON'))
                else:
                    x = '̀'
                    y = '̌'
                if 'ː' not in l[3]:
                    y += 'ː'
                forms[i][3] = l[3].replace(x,y)
        if len([s for s in list(l[2]) if 'GRAVE' in unicodedata.name(s)]) > 0:
            if len([t for t in l[3] if 'GRAVE' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'GRAVE' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'GRAVE' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('GRAVE','CIRCUMFLEX'))
                else:
                    x = '̀'
                    y = '̌'
                forms[i][3] = l[3].replace(x,y)
            elif len([t for t in l[3] if 'ACUTE' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'ACUTE' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'ACUTE' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('ACUTE','CIRCUMFLEX'))
                else:
                    x = '́'
                    y = '̂'
                forms[i][3] = l[3].replace(x,y)
            forms[i][3] = forms[i][3].replace('ː','')
        if len([s for s in list(l[2]) if 'BREVE' in unicodedata.name(s)]) > 0:
            if len([t for t in l[3] if 'ACUTE' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'ACUTE' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'ACUTE' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('ACUTE','CIRCUMFLEX'))
                else:
                    x = '́'
                    y = '̂'
                if 'ː' not in l[3]:
                    y += 'ː'
                forms[i][3] = l[3].replace(x,y)
            elif len([t for t in l[3] if 'CARON' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'CARON' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'CARON' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('CARON','CIRCUMFLEX'))
                else:
                    x = '́'
                    y = '̂'
                if 'ː' not in l[3]:
                    y += 'ː'
                forms[i][3] = l[3].replace(x,y)
            elif len([t for t in l[3] if 'CIRCUMFLEX' in unicodedata.name(t)]) > 0:
                x = [t for t in l[3] if 'CIRCUMFLEX' in unicodedata.name(t)][0]
                if not unicodedata.name(x).startswith('COMBINING'):
                    x = [t for t in l[3] if 'CIRCUMFLEX' in unicodedata.name(t)][0]
                    y = unicodedata.lookup(unicodedata.name(x).replace('CARON','CIRCUMFLEX'))
                else:
                    x = '́'
                    y = '̂'
                if 'ː' not in l[3]:
                    y += 'ː'
                forms[i][3] = l[3].replace(x,y)
        #print(forms[i])



for i,l in enumerate(forms):
    if l[1] == 'slovak':
        forms[i][3] = l[3].replace('tʲ','c')
        forms[i][3] = l[3].replace('lʲj','ʎ')
        forms[i][3] = l[3].replace('lʲ','ʎ')
        forms[i][3] = l[3].replace('lj','ʎ')
        if l[0].endswith('ti') and l[3].endswith('t'):
            forms[i][3] = forms[i][3][:-1]+'c'
        



#polish vowels




#change polish É”

#[l for l in forms if 'ʹ' in l[3]]

#affricates

#search for '.'

#CS
#'old_church_slavic', 'blědъ ʹpaleʹ
#gabati

for i,l in enumerate(forms):
    if l[1].endswith('church_slavic'):
        forms[i][3] = l[3].replace('l̨','ʎ')
        forms[i][3] = l[3].replace('n̨','ɲ')
        forms[i][3] = l[3].replace('r̨','rʲ')
        forms[i][3] = l[3].replace('о̨','ɔ̃')



for i,l in enumerate(forms):
    forms[i][3] = l[3].replace('b͡ʃ','bʃ')
    for x in ['tʂ','dʒ','tʃ','tɕ','ts','dz','tz','dʑ','dʐ']:
        forms[i][3] = l[3].replace(x,x[0]+'͡'+x[1])
        forms[i][3] = l[3].replace('͡͡','͡')


for i,l in enumerate(forms):
    if l[2] == 'tǝnǝ̀k':
        forms[i][3] = 'tǝnə̂k'



f = open('slavic_all_IPA.tsv','w')
for l in forms:
    print('\t'.join(l),file=f)


f.close()


#[l[3][i:i+3] for l in forms for i in range(len(l[3])-2) if l[3][i]=='͡']
[l[3][i:i+3] for l in forms for i in range(len(l[3])-2) if l[3][i:i+3][1]=='͡']



input_segs = sorted(set([s for l in forms for s in l[0]]))
#[' ', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', 'à', 'ā', 'č', 'ē', 'ī', 'ō', 'š', 'ū', 'ž', 'ǫ', '̀', '́', '̄', '̌', '̏', '̑', '̨', 'ъ', 'ь']

output_segs = sorted(set([s for l in forms for s in l[3]]))

