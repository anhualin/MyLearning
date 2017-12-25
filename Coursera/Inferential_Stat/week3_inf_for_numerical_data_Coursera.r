library(statsr)
library(dplyr)
library(ggplot2)

data(nc)
summary(nc$gained)

ggplot(data = nc, aes(x = gained)) +
  geom_histogram(binwidth = 5)

ggplot(data = nc, aes(x = habit, y = weight, color = habit)) +
  geom_boxplot()  
 
nc %>%
  group_by(habit) %>%
  summarise(mean_weight = mean(weight))


table(nc$habit)

inference(y = weight, x = habit, data = nc, statistic = "mean", type = "ci", method = "theoretical")

inference(y = weeks, data = nc, statistic = "mean", type = "ci", conf_level = 0.99, method = "theoretical")

inference(y = weeks, data = nc, statistic = "mean", type = "ci", conf_level = 0.90, method = "theoretical")

inference(y = gained, x = habit, data = nc, statistic = "mean", type = "ht", 
          null = 0, alternative="twosided",
          method ="theoretical")
