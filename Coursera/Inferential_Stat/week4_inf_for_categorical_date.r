install.packages("devtools")
library(devtools)
install_github("StatsWithR/statsr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("shiny")

install.packages("rmarkdown")

library(statsr)
library(dplyr)
library(ggplot2)
data(atheism)

us12 <- atheism %>%
  filter(nationality == 'United States', year ==  2012 )

us12 %>%
  summarise(athe_mean = mean(as.numeric(response) - 1))

inference(y = response, data = us12, statistic = 'proportion',
          type = 'ci', method = 'theoretical', success = 'atheist')

spain <- atheism %>%
  filter(nationality == 'Spain')

inference(y = response, x = as.factor(year), data = spain, statistic = 'proportion',
          type = 'ht', method = 'theoretical', alternative = 'twosided',
          success = 'atheist')

p_pool = (115 + 103) /(2291)
SE = sqrt(p_pool * (1 - p_pool) *(1/1146 + 1/1145))

p_dif_hat = 115/1146 - 103/1145

Z <- p_dif_hat/SE

p_value <- pnorm(Z, lower.tail = FALSE) * 2
