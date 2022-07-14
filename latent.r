#### Polya tree
library(usedist)
library(reshape2)

data <- read.table('~/Downloads/osr.txt', header=T)
colnames(data)[2:3] <- c("n+", "n-")
# split each row into two

data <- as.matrix(data)
df <- data[1, c(1,2,4:6)]
df <- rbind(df, data[1,c(1,3,7:9)])

for(i in 3:nrow(data)) {
    df <- rbind(df, data[i, c(1, 2, 4:6)])
    df <- rbind(df, data[i, c(1, 3, 7:9)])
}
df <- cbind(df, rep(c("+","-"), nrow(df)/2))
colnames(df)[c(2, 6)] <- c("n", "bio")
df <- as.data.frame(df)
df[,2:5] <- apply(df[,2:5], 2, as.numeric)
#df[,c(1,6)] <- apply(df[,c(1,6)], 2, as.factor) 

ham_dist <- function(x,y){
    sum( x[] != y)
}

hclust(dist_make(data, ham_dist))