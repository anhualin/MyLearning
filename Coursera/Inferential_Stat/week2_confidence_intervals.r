set.seed(12345)
library(statsr)
library(dplyr)
library(ggplot2)

data(ames)

n <- 60
samp <- ames %>%
  sample_n(n)

ggplot(data = samp, aes(x = area)) + 
  geom_histogram(binwidth = 250)

z_star_95 <- qnorm(0.975)

samp %>%
  summarise(lower = mean(area) - z_star_95 * (sd(area) / sqrt(n)),
            upper = mean(area) + z_star_95 * (sd(area) / sqrt(n)))

params <- ames %>%
  summarise(mu = mean(area))

ci <- ames %>%
  rep_sample_n(size = n, reps = 50, replace = TRUE) %>%
  summarise(lower = mean(area) - z_star_95 * (sd(area) / sqrt(n)),
            upper = mean(area) + z_star_95 * (sd(area) / sqrt(n)))

ci <- ci %>%
  mutate(capture_mu = ifelse(lower < params$mu & upper > params$mu, 'yes', 'no'))

ci_data <- data.frame(ci_id = c(1:50, 1:50),
                      ci_bounds = c(ci$lower, ci$upper),
                      capture_mu = c(ci$capture_mu, ci$capture_mu))

ggplot(data = ci_data, aes(x = ci_bounds, y = ci_id, 
                          group = ci_id, color = capture_mu)) +
  geom_point(size = 2) +  # add points at the ends, size = 2
  geom_line() +           # connect with lines
  geom_vline(xintercept = params$mu, color = "darkgray") # draw vertical line