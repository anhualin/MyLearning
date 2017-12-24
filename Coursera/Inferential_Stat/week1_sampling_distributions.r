library(statsr)
library(dplyr)
library(shiny)
library(ggplot2)

data(ames)

ggplot(data = ames, aes(x = area)) +
  geom_histogram(binwidth = 250)

ames %>%
  summarise(mu = mean(area), pop_med = median(area),
            sigma = sd(area), pop_iqr = IQR(area),
            pop_min = min(area), pop_max = max(area),
            pop_q1 = quantile(area, 0.25),
            pop_q3 = quantile(area, 0.75))

samp1 <- ames %>%
  sample_n(size = 50)


ggplot(data = samp1, aes(x = area)) + 
  geom_histogram(binwidth = 250)

samp1 %>%
  summarise(mu_s = mean(area), pop_med_s = median(area),
            sigma_s = sd(area), pop_iqr_s = IQR(area),
            pop_min_s = min(area), pop_max_s = max(area),
            pop_q1_s = quantile(area, 0.25),
            pop_q3_s = quantile(area, 0.75))


samp2 <- ames %>%
  sample_n(size = 100)

samp3 <- ames %>%
  sample_n(size = 1000)

samp1 %>%
  summarise(x_bar = mean(area))

samp2 %>%
  summarise(x_bar = mean(area))

samp3 %>%
  summarise(x_bar = mean(area))

ames %>%
  sample_n(size = 50) %>%
  summarise(x_bar = mean(area))

sample_means50 <- ames %>%
  rep_sample_n(size = 50, reps = 15000, replace = TRUE) %>%
  summarise(x_bar = mean(area))

ggplot(data = sample_means50, aes(x = x_bar)) +
  geom_histogram(binwidth = 20)

sample_means_small <- ames %>%
  rep_sample_n(size = 10, reps = 25, replace = TRUE) %>%
  summarise(x_bar = mean(area))

samp1 <- ames %>%
  sample_n(size = 50)

samp1 %>%
  summarise(x_bar = mean(price))

sample_means50 <- ames %>%
  rep_sample_n(size = 50, reps = 5000, replace = TRUE) %>%
  summarise(x_bar = mean(price))

ggplot(data = sample_means50, aes(x = x_bar)) + 
  geom_histogram(binwidth = 2000)

sample_means150 <- ames %>%
  rep_sample_n(size = 150, reps = 5000, replace = TRUE) %>%
  summarise(x_bar = mean(price))

ggplot(data = sample_means150, aes(x = x_bar)) + 
  geom_histogram(binwidth = 2000)