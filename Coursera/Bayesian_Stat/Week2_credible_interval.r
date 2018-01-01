library(statsr)
library(dplyr)
library(ggplot2)
library(shiny)
library(statsr)
data(brfss)

table(brfss$sex)

n <- length(brfss$sex)
x <- sum(brfss$sex == 'Female')

alpha <- 1 + 3868
beta <- 1 + 1132
lb <- 0.758
ub <- 0.789
pbeta(ub, alpha, beta) - pbeta(lb, alpha, beta)

pbeta(0.5, 5030, 4972)

dbinom(5029, 10000, 0.5)*10001
