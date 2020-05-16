require(phytools)


reference <- read.newick('slavic_reference.tre')
reference$edge.length <- NULL

#pdf('reference.pdf')
#plot(reference,cex=.75,font=1,type='u')
#dev.off()

ST.data <- read.delim('ST_embeddings.txt',sep=' ',row.names=1,header=F)
ST <- nj(dist(ST.data))
ST$edge.length <- NULL

#pdf('ST.pdf')
#plot(ST,cex=.75,font=1,type='u')
#dev.off()

sigmoid.data <- read.delim('sigmoid_embeddings.txt',sep=' ',row.names=1,header=F)
sigmoid <- nj(dist(sigmoid.data))
sigmoid$edge.length <- NULL

#pdf('sigmoid.pdf')
#plot(sigmoid,cex=.75,font=1,type='u')
#dev.off()

dense.data <- read.delim('dense_embeddings.txt',sep=' ',row.names=1,header=F)
dense <- nj(dist(dense.data))
dense$edge.length <- NULL

#pdf('dense.pdf')
#plot(dense,cex=.75,font=1,type='u')
#dev.off()


pdf('trees.pdf')

par(mfrow=c(2,2),mar=c(0,0,0,0))

plot(reference,font=1,type='u')#,mar=c(0,0,0,0))
plot(dense,font=1,type='u')#,mar=c(0,0,0,0))
plot(ST,font=1,type='u')#,mar=c(0,0,0,0))
plot(sigmoid,font=1,type='u')#,mar=c(0,0,0,0))

dev.off()



#QuartetDistance('ST.tre','slavic_reference.tre')