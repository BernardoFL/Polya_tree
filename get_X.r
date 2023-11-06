pfsr <- read.csv("~/Desktop/pfsr.csv") 
pfsr <- pfsr[-13,]
pfsr <- pfsr[(pfsr$lo < pfsr$median) & (pfsr$median < pfsr$hi),]
pfsr <- pfsr[-(30:31), ]
pfsr$k <- as.factor(pfsr$k)
levels(pfsr$k) <- 1:length(levels(pfsr$k))
pfsr$bio <- as.factor(pfsr$bio)
pfsr$tumor <- as.factor(pfsr$tumor)
pfsr$offset <- as.factor(1:nrow(pfsr))

write.csv(cbind(pfsr$bio, pfsr$k, pfsr$n, pfsr$lo, pfsr$median, pfsr$hi, model.matrix(~offset + tumor + k -1, data=pfsr)), "~/Documents/X.csv")
library(ggplot2)
# Load the ggplot2 library


# Example data #el probelma es que hay 34 rf 
bins <- c(1, 3, 4, 5, 6)
heights <- c(10, 20, 15, 30)

# Create a data frame with bins and heights
data <- data.frame(BinStart = bins[-length(bins)], BinEnd = bins[-1], Heights = heights)

# Create the histogram using ggplot
 ggplot(data, aes(x = factor(BinStart), y = Heights, fill = factor(BinStart))) +
    geom_bar(stat = "identity") +
    labs(title = "Histogram", x = "Bins", y = "Heights") +
    theme_minimal() +
    scale_x_discrete(labels = paste0(data$BinStart, "-", data$BinEnd))
