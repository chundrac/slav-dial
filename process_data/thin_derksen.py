text=[l.strip().split('\t') for l in open('derksen_uncleaned_data.tsv','r')]

langdict = {
    'Belorussian:':'belarusian',
    'Bulgarian:':'bulgarian',
    'Church Slavic:':'church_slavic',
    'Czech:':'czech',
    'Lower Sorbian:':'lower_sorbian',
    #'Macedonian:':'',
    'Old Church Slavic:':'old_church_slavic',
    'Polish:':'polish',
    'Russian:':'russian',
    'Serbo-Croatian:':'bcs',
    'Slovak:':'slovak',
    'Slovene:':'slovene',
    'Ukrainian:':'ukrainian',
    'Upper Sorbian:':'upper_sorbian'
    }

new_text = []
for l in text:
    ll = l
    if l[3] in langdict.keys():
        ll[3] = langdict[ll[3]]
        new_text.append(ll)



new_text = sorted(set([tuple(l) for l in new_text if len(l) == 7]))

f = open('derksen_slavic.tsv','w')
for l in new_text:
    print('\t'.join(l),file=f)


f.close()