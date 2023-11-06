########## Basic metanalysis

library(metamedian)
library(metafor)
library(dplyr)

data <- read.csv("~/Desktop/pfsr_with_cov.csv")
data <- data[-13, ]
data <- data[(data$lo. < data$median.) & (data$median. < data$hi.), ]
data <- data[(data$lo..1 < data$median..1) & (data$median..1 < data$hi..1), ]
data <- data[-(30:31), ]
data$k <- as.factor(data$k)
levels(data$k) <- 1:length(levels(data$k))
data$k <- as.character(data$k)
names(data)[2:9] <- c(
    "n.g1", "n.g2", "med.g1", "q1.g1",
    "q3.g1", "med.g2", "q1.g2", "q3.g2"
)
# remove the 11th and 165th obs
data <- data[-c(7, 11, 18, 30, 40), ]
## filter data without confidence intervals. Estimate mean and sd from the Wan et al (2014)
data <- data %>% filter((q1.g1 != 0 & q3.g1 != 0) | (q1.g2 != 0 & q3.g2 != 0)) %>% 
    mutate(mean.g1 = (q1.g1 + med.g1 + q3.g1)/3, sd.g1 = (q3.g1 - q1.g1)/qnorm((0.75*n.g1 -0.125 )/(n.g1 + 0.25)),
    mean.g2 = (q1.g2 + med.g2 + q3.g2)/3, sd.g2 = (q3.g2 - q1.g2)/qnorm((0.75*n.g2 -0.125 )/(n.g2 + 0.25)))

data <- data %>% filter(n.g1 > 3 & n.g2 > 3)
data <- data[-c(7, 27), ]
data <- data[-10,]
res <- escalc(m1i = mean.g1, m2i = mean.g2, n1i = n.g1, n2i = n.g2, sd1i = sd.g1, sd2i = sd.g2, data=data, measure="ROM")

png("~/Documents/Code/Polya_tree/forest_mean.png")
forest(rma(yi, vi, data=res), annotate=F)
dev.off()


with(res, forest(yi,
    ci.lb = log(ci.lb), ci.ub = log(ci.ub), header = TRUE, atransf = exp,
    xlim = c(-8, 6), at = log(c(0.05, 0.25, 1, 4, 16)),
    psize = 1, ylim = c(-1.5, res$k + 3)
))
## now try using metamedian
median_anal <- metamedian(data, median_method = "qe")
summary(median_anal)

png("~/Documents/Code/Polya_tree/forest_median.png")
forest(median_anal, annotate=T, xlim = c(-10,10))
dev.off()

####
data_est <- read.csv("~/Documents/Code/Polya_tree/inter_res.csv")
with(data_est, forest(x = x2, ci.lb = x1, ci.ub = x3, annotate=F))


