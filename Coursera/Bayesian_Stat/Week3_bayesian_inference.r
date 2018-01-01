library(statsr)
library(dplyr)
library(ggplot2)

data(nc)

summary(nc$weight)
ggplot(data = nc, aes(x = weight)) + 
  geom_histogram(binwidth = 0.5)

bayes_inference(y = weight, data = nc, statistic = "mean", type = "ci", cred_level = 0.99)

ggplot(data = nc, aes(x = habit, y = weight, color = habit)) +
  geom_boxplot()  