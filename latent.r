#### Polya tree
library(usedist)
library(mvtnorm)

data <- read.table('~/Downloads/osr.txt', header=T)
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

# This gets the l2 product metric using the Hamming distance for k and the
# biomarker and the l2 metric for the quantiles and sample size.
# TO TRY: weight the distances

prod_dist <- function(x,y) {
   dist(rbind(sum( x[c(1,6)] != y[c(1,6)]), dist(rbind(x[2:5], y[2:5]))))
}



# Create tree and obtain a matrix of assignments
tree <- hclust(dist_make(na.pass(df), prod_dist))
clust <- cutree(tree, 1:nrow(df))

# Generate random effects
pesos <- matrix(0, nrow(df), 2)
for(k in seq(1, nrow(df)-1, 2)) {
    sigma <- rbind(c(30/log(k+1), 0), c(0, 20/log(k+1)))
    zkm_1 <- rmvnorm(k, sigma=sigma,  checkSymmetry = F)
    zkm_2 <- rmvnorm(k+1, sigma = sigma, checkSymmetry = F)
    pesos[k,] <- pesos[k,] + zkm_1[clust[c(k,k+1),k]] + zkm_2[clust[c(k,k+1),k+1]] #where are the biomarker pairs in clust
}

