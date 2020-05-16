from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

ST = []
for i in range(10):
  for l in open('decoded_ST_{}.tsv'.format(i),'r'):
    ST.append(l.strip().split('\t'))

dense = []
for i in range(10):
  for l in open('decoded_dense_{}.tsv'.format(i),'r'):
    dense.append(l.strip().split('\t'))

sigmoid = []
for i in range(10):
  for l in open('decoded_sigmoid_{}.tsv'.format(i),'r'):
    sigmoid.append(l.strip().split('\t'))


denseLD = [float(l[-1]) for l in dense]
STLD = [float(l[-1]) for l in ST]
sigmoidLD = [float(l[-1]) for l in sigmoid]

countdict = defaultdict(int)
distdict = defaultdict(list)

for l in dense:
  countdict[l[0]]+=1

for l in dense:
  distdict[l[0]].append(float(l[-1]))

for k in distdict.keys():
  distdict[k]=np.mean(distdict[k])

for k in distdict.keys():
  plt.plot(np.log(countdict[k]),np.log(distdict[k]),'.',c='#1f77b4')
  plt.annotate(xy=(np.log(countdict[k]),np.log(distdict[k])),s=k)


countdict = defaultdict(int)
distdict = defaultdict(list)

for l in ST:
  countdict[l[0]]+=1

for l in ST:
  distdict[l[0]].append(float(l[-1]))

for k in distdict.keys():
  distdict[k]=np.mean(distdict[k])

for k in distdict.keys():
  plt.plot(np.log(countdict[k]),distdict[k],'.',c='#1f77b4')
  plt.annotate(xy=(np.log(countdict[k]),distdict[k]),s=k)


plt.savefig('dense_freq.pdf')
plt.clf()

countdict = defaultdict(int)
distdict = defaultdict(list)

for l in sigmoid:
  countdict[l[0]]+=1

for l in sigmoid:
  distdict[l[0]].append(float(l[-1]))

for k in distdict.keys():
  distdict[k]=np.mean(distdict[k])

for k in distdict.keys():
  plt.plot(np.log(countdict[k]),distdict[k],'.',c='#1f77b4')
  plt.annotate(xy=(np.log(countdict[k]),distdict[k]),s=k)


plt.savefig('sigmoid_freq.pdf')
plt.clf()

countdict = defaultdict(int)
distdict = defaultdict(list)

for l in ST:
  countdict[l[0]]+=1

for l in ST:
  distdict[l[0]].append(float(l[-1]))

for k in distdict.keys():
  distdict[k]=np.mean(distdict[k])

for k in distdict.keys():
  plt.plot(np.log(countdict[k]),distdict[k],'.',c='#1f77b4')
  plt.annotate(xy=(np.log(countdict[k]),distdict[k]),s=k)


plt.savefig('ST_freq.pdf')