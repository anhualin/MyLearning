subject <- factor(rep(1:6, each = 3))
score <- factor(rep(1:3, times = 6))
y <- c(10, 11, 12, 10, 12, 14, 12, 13, 14, 12, 14, 16, 14, 15,
       16, 14, 16, 18)
Data <- data.frame(subject, score, y)
with(Data, tapply(y, list(subject = subject), mean))

# treat subject as fixed factor
fit.lm <- lm(y ~ 1 + subject, data = Data)
anova(fit.lm)

# treat subject as random factor
fit.aov <- aov(y ~ 1 + Error(subject), data = Data)
summary(fit.aov)

# modern approach treating subject as random factor
library(lme4)
fit.lmer <- lmer(y ~ 1 + (1 | subject), data = Data)
summary(fit.lmer)
