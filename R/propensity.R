#
# R SCRIPT FOR ANALYSIS OF EXAMPLE DATASET BY REGRESSION ADJUSTMENT AND PROPENSITY SCORES
# Ben Cowling - April 30, 2008
#

#
# first, read in the data ...

example <- read.csv("propensity.csv", header=TRUE)

#
# load additional packages required for some plots:

require(Hmisc)       # (Frank Harrell's library of useful functions)
require(mgcv)        # (non-linear regression functions)
require(boot)        # (bootstrapping functions)
require(Matching)    # (propensity score matching functions)

#
# SECTION 3 - DESCRIPTIVE ANALYSIS ...
#

#
# check the 2x2 table of 30-day mortality vs treatment

table(example$death, example$trt)

#
# check the odds ratio for the effect of treatment on 30-day mortality

model1 <- glm(death~trt, family="binomial", data=example)
coefs <- model1$coef             # parameter estimates
se <- sqrt(diag(vcov(model1)))   # parameter standard errors
exp(coefs)[2]                    # estimated odds ratio
exp(coefs - 1.96*se)[2]          # lower limit of 95% CI
exp(coefs + 1.96*se)[2]          # upper limit of 95% CI

#
# check the absolute difference in mortality rates

abs.diff <- prop.test(c(30,40), c(192,208))
abs.diff$estimate[1] - abs.diff$estimate[2]
abs.diff$conf.int

#
# plot the distribution of explanatory variables between treatment groups

par(mar=c(5,5.5,1,1), lheight=1.5)
out1 <- histbackback(split(example$age, example$trt), brks=39.5+0:31, xlim=c(-16.5, 16.5), axes=FALSE)
axis(1)
axis(2, las=1, at=0:6*5+0.5, labels=40+0:6*5)
mtext(c("trt=0", "trt=1"), side=1, at=c(-7.5, 7.5), line=3)
mtext("Age", side=2, at=28, line=2.4, las=1)
barplot(-out1$left, col="red" , horiz=TRUE, space=0, add=TRUE, axes=FALSE)
barplot(out1$right, col="blue", horiz=TRUE, space=0, add=TRUE, axes=FALSE)

out2 <- histbackback(split(example$risk, example$trt), brks=0:6-0.5, xlim=c(-80.5, 80.5), axes=FALSE)
axis(1)
axis(2, las=1, at=1:6-0.5, labels=0:5)
mtext(c("trt=0", "trt=1"), side=1, at=c(-40, 40), line=3)
mtext("Risk
score", side=2, at=5.5, line=2.4, las=1)
barplot(-out2$left, col="red" , horiz=TRUE, space=0, add=TRUE, axes=FALSE)
barplot(out2$right, col="blue", horiz=TRUE, space=0, add=TRUE, axes=FALSE)

out2 <- histbackback(split(example$severity, example$trt), brks=0:11-0.5, xlim=c(-50.5, 50.5), axes=FALSE)
axis(1)
axis(2, las=1, at=2*0:5+0.5, labels=0:5*2)
mtext(c("trt=0", "trt=1"), side=1, at=c(-25, 25), line=3)
mtext("Severity
index", side=2, at=9.5, line=2.4, las=1)
barplot(-out2$left, col="red" , horiz=TRUE, space=0, add=TRUE, axes=FALSE)
barplot(out2$right, col="blue", horiz=TRUE, space=0, add=TRUE, axes=FALSE)

#
# apply t.test and chisq.test for formal comparison

t.test(age~trt, data=example)
chisq.test(table(example$risk, example$trt))
t.test(severity~trt, data=example)


#
# SECTION 4 - LOGISTIC REGRESSION ...
#

#
# for M1
table(example$trt)
m1 <- glm(death ~ trt, family="binomial", data=example)
coefs <- m1$coef             # parameter estimates
se <- sqrt(diag(vcov(m1)))   # parameter standard errors
exp(coefs)[2]                # estimated odds ratio
exp(coefs - 1.96*se)[2]      # lower limit of 95% CI
exp(coefs + 1.96*se)[2]      # upper limit of 95% CI

#
# for M2 categorise age into agegp first
example$agegp <- cut(example$age, breaks=c(40, 50, 60, 70), right=FALSE, include.lowest=TRUE)
table(example$agegp)
m2 <- glm(death ~ trt + agegp, family="binomial", data=example)
coefs <- m2$coef             # parameter estimates
se <- sqrt(diag(vcov(m2)))   # parameter standard errors
exp(coefs)[-1]               # estimated odds ratio
exp(coefs - 1.96*se)[-1]     # lower limit of 95% CI
exp(coefs + 1.96*se)[-1]     # upper limit of 95% CI

#
# for M3 categorise risk and severity
example$riskgp <- cut(example$risk, breaks=c(0, 1, 2, 4, 5), right=FALSE, include.lowest=TRUE)
table(example$risk)
table(example$riskgp)
example$sevgp <- cut(example$severity, breaks=c(0, 4, 5, 6, 7, 10), right=FALSE, include.lowest=TRUE)
table(example$severity)
table(example$sevgp)
m3 <- glm(death ~ trt + agegp + riskgp + sevgp, family="binomial", data=example)
coefs <- m3$coef             # parameter estimates
se <- sqrt(diag(vcov(m3)))   # parameter standard errors
exp(coefs)[-1]               # estimated odds ratio
exp(coefs - 1.96*se)[-1]     # lower limit of 95% CI
exp(coefs + 1.96*se)[-1]     # upper limit of 95% CI

AIC(m1,m2,m3)

#
# for M4 use linear effects
m4 <- glm(death ~ trt + age + risk + severity, family="binomial", data=example)
coefs <- m4$coef             # parameter estimates
se <- sqrt(diag(vcov(m4)))   # parameter standard errors
exp(coefs)[-1]               # estimated odds ratio
exp(coefs - 1.96*se)[-1]     # lower limit of 95% CI
exp(coefs + 1.96*se)[-1]     # upper limit of 95% CI

AIC(m1,m2,m3,m4)

#
# for m5 use non-linear regression
m5 <- gam(death ~ trt + s(age) + s(severity), family="binomial", data=example)
plot.gam(m5, select=1, rug=F)
plot.gam(m5, select=2, rug=F)

AIC(m5)


#
# SECTION 5 - PROPENSITY SCORES ...
#


m.ps <- glm(trt ~ age + risk + severity, family="binomial", data=example)
# store the predicted probabilities ie propensity scores
example$ps <- predict(m.ps, type="response")
summary(example$ps)

#
# Stratification into quintiles

example$psgp <- cut(example$ps, breaks=quantile(example$ps, prob=0:5*0.2),
 labels=c("Q1","Q2","Q3","Q4","Q5"), right=FALSE, include.lowest=TRUE)

#
# Check the distribution of propensity scores between treatment groups

plot(example$ps, jitter(example$trt+0.5), pch=16, cex=0.5, col=2*example$trt+2,
 axes=FALSE, type="p", xlim=c(0,1), ylim=c(0,2), xlab="", ylab="")
axis(1)
mtext(c("trt=0", "trt=1"), side=2, at=c(0.5, 1.5), line=3, las=1, adj=0)
mtext("Propensity score", side=1, at=0.5, line=2.5)
lines(quantile(example$ps[example$trt==0])[c(2,4)], c(0.85, 0.85), col=2)
lines(rep(quantile(example$ps[example$trt==0])[3], 2), c(0.8, 0.9), col=2)
lines(quantile(example$ps[example$trt==1])[c(2,4)], c(1.15, 1.15), col=4)
lines(rep(quantile(example$ps[example$trt==1])[3], 2), c(1.1, 1.2), col=4)

#
# Check whether the propensity score stratification has balanced /age/

plot(0, type="n", axes=FALSE, xlim=c(0.75,6.25), ylim=c(40,70), xlab="", ylab="")
axis(1, at=1:5, labels=paste("Q", 1:5, sep=""))
axis(1, at=6, labels="")
lines(c(5.75,6.25), rep(40,2))
axis(2, las=1, at=4:7*10)
mtext("Age", side=2, at=65, line=4.5, las=1, adj=0)
mtext("Quintiles of propensity score", side=1, at=3, line=2.5)
mtext("Overall", side=1, at=6, line=1)
lines(rep(6-0.05, 2), quantile(example$age[example$trt==0])[c(2,4)], col=2)
lines(rep(6+0.05, 2), quantile(example$age[example$trt==1])[c(2,4)], col=4)
lines(c(6-0.08,6-0.02), rep(quantile(example$age[example$trt==0])[c(3)],2), col=2)
lines(c(6+0.02,6+0.08), rep(quantile(example$age[example$trt==1])[c(3)],2), col=4)
for(i in 1:5){
  lines(rep(i-0.05, 2), quantile(example$age[example$trt==0 & example$psgp==paste("Q", i, sep="")])[c(2,4)], col=2)
  lines(rep(i+0.05, 2), quantile(example$age[example$trt==1 & example$psgp==paste("Q", i, sep="")])[c(2,4)], col=4)
  lines(c(i-0.08,i-0.02), rep(quantile(example$age[example$trt==0 & example$psgp==paste("Q", i, sep="")])[c(3)],2), col=2)
  lines(c(i+0.02,i+0.08), rep(quantile(example$age[example$trt==1 & example$psgp==paste("Q", i, sep="")])[c(3)],2), col=4)
}

#
# Check whether the propensity score strata balance all explanatory variables

age.unbal <- lm(age~trt, data=example)
risk.unbal <- lm(risk~trt, data=example)
severity.unbal <- lm(severity~trt, data=example)

age.bal <- lm(age~trt+psgp, data=example)
risk.bal <- lm(risk~trt+psgp, data=example)
severity.bal <- lm(severity~trt+psgp, data=example)

#pull out the t-values
summary(age.unbal)
summary(age.unbal)$coefficients[2,3]   # extract the relevant t-value
balance <- data.frame(before=rep(NA, 3), after=rep(NA, 3), row.names=c("age", "risk", "severity"))
balance[1,1] <- summary(age.unbal)$coefficients[2,3]
balance[2,1] <- summary(risk.unbal)$coefficients[2,3]
balance[3,1] <- summary(severity.unbal)$coefficients[2,3]
balance[1,2] <- summary(age.bal)$coefficients[2,3]
balance[2,2] <- summary(risk.bal)$coefficients[2,3]
balance[3,2] <- summary(severity.bal)$coefficients[2,3]
balance                       # print the t statistics

#
# plot absolute t-statistics
plot(0, type="n", axes=FALSE, xlim=c(0,5), ylim=c(0.75,3.25), xlab="", ylab="")
axis(1)
lines(c(0,0), c(0.75,3.25))
mtext(c("Age", "Risk", "Severity"), side=2, at=3:1, line=4.5, las=1, adj=0)
mtext(expression(paste("Absolute ", italic(t), "-statistic", sep="")), side=1, at=2.5, line=2.5)
points(abs(balance)[3:1,1], 3:1, pch=16)
points(abs(balance)[3:1,2], 3:1, pch= 1)


#
# STRATIFICATION INTO QUINTILES

death.QT0 <- split(example$death[example$trt==0], example$psgp[example$trt==0])
death.QT1 <- split(example$death[example$trt==1], example$psgp[example$trt==1])
sapply(death.QT1, mean)                                # estimates for group trt=1
sapply(death.QT0, mean)                                # estimates for group trt=0
sapply(death.QT1, mean)-sapply(death.QT0, mean)        # treatment effect

# plot death rates by propensity score quintile
plot(0, type="n", axes=FALSE, xlim=c(0.75,5.25), ylim=c(0,0.5), xlab="", ylab="")
axis(1, at=1:5, labels=paste("Q", 1:5, sep=""))
axis(2, las=1, at=0:5/10, labels=paste(0:5*10, "%", sep=""))
mtext("30-day mortality rate", side=2, line=2)
mtext("Quintiles of propensity score", side=1, at=3, line=2.5)
for(i in 1:5){
  lines(rep(i-0.05, 2), binom.test(sum(death.QT0[[i]]), length(death.QT0[[i]]))$conf.int, col=2)
  lines(rep(i+0.05, 2), binom.test(sum(death.QT1[[i]]), length(death.QT1[[i]]))$conf.int, col=4)
  points(i-0.05, mean(death.QT0[[i]]), pch=16, cex=0.8, col=2)
  points(i+0.05, mean(death.QT1[[i]]), pch=16, cex=0.8, col=4)
}

# apply prop.test to each quintile for formal comparison
psgp.summary <- as.matrix(data.frame(nT0=summary(example$psgp[example$trt==0]),
  nT1=summary(example$psgp[example$trt==1]),
  dT0=summary(example$psgp[example$trt==0 & example$death==1]),
  dT1=summary(example$psgp[example$trt==1 & example$death==1]),
  row.names=levels(example$psgp)))
PS.s <- list(NA)
PS.s.se <- PS.s.estimates <- rep(NA, 5)
for (i in 1:5){
  PS.s[[i]] <- prop.test(x=psgp.summary[i,c("dT1", "dT0")], 
    n=psgp.summary[i,c("nT1", "nT0")])
  PS.s.estimates[i] <- PS.s[[i]]$estimate[1] - PS.s[[i]]$estimate[2]
  PS.s.se[i] <- abs(PS.s[[i]]$conf.int[2] - PS.s[[i]]$conf.int[1])/1.96/2
}
PS.s.overall.mean <- weighted.mean(x=PS.s.estimates, w=rowSums(psgp.summary[,c("nT1", "nT0")]))
PS.s.overall.se <- sqrt(weighted.mean(x=PS.s.se*PS.s.se, w=rowSums(psgp.summary[,c("nT1", "nT0")])))

PS.s.overall.mean                                    # overall mean
PS.s.overall.mean + c(-1.96,1.96)*PS.s.overall.se    # 95% confidence interval

#
# plot estimates by propensity score quintile, and overall estimate
plot(0, type="n", axes=FALSE, xlim=c(0.75,6.25), ylim=c(-0.4,0.4), xlab="", ylab="")
axis(1, at=1:5, labels=paste("Q", 1:5, sep=""))
axis(1, at=6, labels="")
lines(c(5.75,6.25), rep(-0.4,2))
axis(2, las=1, at=-2:2/5, labels=paste(-2:2*20, "%", sep=""))
lines(c(0.75,6.25), c(0,0), col=gray(0.8))
mtext("Absolute difference in mortality rates", side=2, line=2)
mtext("Quintiles of propensity score", side=1, at=3, line=2.5)
mtext("Overall", side=1, at=6, line=1)
for(i in 1:5){
  lines(rep(i, 2), PSs[[i]]$conf.int)
  points(i, PSs[[i]]$estimate[1] - PSs[[i]]$estimate[2], pch=16, cex=0.8, col=1)
}
lines(rep(6, 2), PSs.overall.mean + c(-1.96,1.96)*PSs.overall.se)
points(6, PSs.overall.mean, pch=18, cex=1.5, col=1)


#
# MATCHING BY PROPENSITY SCORES

require(Matching)
PS.m <- Match(Y=example$death, Tr=example$trt, X=example$ps, M=1, caliper=0.2, replace=FALSE)
MatchBalance(death~age+risk+severity, data=example,
  match.out=PS.m)                     # check that the PS matching has achieved balance

PS.m$est                              # estimate of the treatment effect
PS.m$est - 1.96*PS.m$se               # lower limit of 95% CI
PS.m$est + 1.96*PS.m$se               # upper limit of 95% CI

PS.m.d0 <- sum(PS.m$mdata$Y[PS.m$mdata$Tr==0])
PS.m.d1 <- sum(PS.m$mdata$Y[PS.m$mdata$Tr==1])
PS.m.n0 <- length(PS.m$mdata$Y[PS.m$mdata$Tr==0])
PS.m.n1 <- length(PS.m$mdata$Y[PS.m$mdata$Tr==1])

PS.m.OR <- (PS.m.d1/(PS.m.n1-PS.m.d1)) / (PS.m.d0/(PS.m.n0-PS.m.d0))
PS.m.logOR.se <- sqrt(1/PS.m.d1 + 1/(PS.m.n1-PS.m.d1) + 1/PS.m.d0 + 1/(PS.m.n0-PS.m.d0))
PS.m.OR                                # (naive) estimated odds ratio ( - NOT ADJUSTED FOR PS MATCHING)
exp(log(PS.m.OR) - 1.96*PS.m.logOR.se) # lower limit of 95% CI for OR
exp(log(PS.m.OR) + 1.96*PS.m.logOR.se) # upper limit of 95% CI for OR


#
# INVERSE WEIGHTING BY PROPENSITY SCORES

PS.w.T0 <- mean(example$death*(example$trt==0)/(1-example$ps))
PS.w.T1 <- mean(example$death*(example$trt==1)/example$ps)
PS.w.T1 - PS.w.T0     # treatment effect

# use bootstrapping to derive the 95% CI
require(boot)
fn.invwt.abs <- function(data, indices){
  newdata <- data[indices,]
  w.T0 <- mean(newdata$death*(newdata$trt==0)/(1-newdata$ps))
  w.T1 <- mean(newdata$death*(newdata$trt==1)/newdata$ps)
  w.T1 - w.T0
}
boot.wt.abs <- boot(example, fn.invwt.abs, R=1000, stype="i")
quantile(boot.wt.abs$t, c(0.025, 0.5, 0.975))

# also look at the odds ratio
PS.w.T0 <- mean(example$death*(example$trt==0)/(1-example$ps))
PS.w.T1 <- mean(example$death*(example$trt==1)/example$ps)
( (PS.w.T1)/(1-PS.w.T1) ) / ( (PS.w.T0)/(1-PS.w.T0) )

# use bootstrapping to derive the 95% CI
require(boot)
fn.invwt.or <- function(data, indices){
  newdata <- data[indices,]
  w.T0 <- mean(newdata$death*(newdata$trt==0)/(1-newdata$ps))
  w.T1 <- mean(newdata$death*(newdata$trt==1)/newdata$ps)
  ( (w.T1)/(1-w.T1) ) / ( (w.T0)/(1-w.T0) )
}
boot.wt.or <- boot(example, fn.invwt.or, R=1000, stype="i")
quantile(boot.wt.or$t, c(0.025, 0.5, 0.975))


#
# LOGISTIC REGRESSION ADJUSTING FOR THE PROPENSITY SCORES

PS.l <- glm(death ~ trt + psgp, family="binomial", data=example)
PS.l.se <- sqrt(diag(vcov(PS.l)))     # parameter standard errors
exp(PS.l$coef)[2]                     # estimated odds ratio
exp(PS.l$coef - 1.96*PS.l.se)[2]      # lower limit of 95% CI
exp(PS.l$coef + 1.96*PS.l.se)[2]      # upper limit of 95% CI









#
# the end
#
#
