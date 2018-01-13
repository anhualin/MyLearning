
dat <- read.csv(file = 'T:\\RNA\\Baltimore\\alin\\Trunk\\dat.csv')
set.seed(12345)
dat$tc <- rbinom(nrow(dat), 1, 0.4)
dat$ret <- rbinom(nrow(dat), 1, 0.7)
dat$prog <- rbinom(nrow(dat), 1, 0.6)
dat$balance <- dat$APF_RES_Balance
dat$honesty <- dat$Academic_Honesty_Declaration
dat$pay_plan <- dat$Payment_Plan_Type
dat$assignment <- dat$Turnitin
dat$region <- dat$Region
dat$forum <- dat$Total_posts

dat$Program <- NULL
dat$Module <- NULL
dat$APF_RES_Balance <- NULL
dat$WorldPay_Completed <- NULL
dat$Payment_Plan_Type <- NULL
dat$Academic_Honesty_Declaration <- NULL
dat$Forum_W1 <- NULL
dat$Region <- NULL
dat$Country <- NULL
dat$Total_posts <- NULL
dat$Turnitin <- NULL
dat$Forum_W2 <- NULL
dat$Paid <- NULL

dat$pay_plan <- ifelse(dat$pay_plan == 'Module/Pay As You Go', 1, 0)
dat$africa <- ifelse(dat$region == 'AFRICA', 1, 0)
dat$asia <- ifelse(dat$region == 'ASIA', 1, 0)
dat$europe <- ifelse(dat$region == 'EUROPE', 1, 0)
dat$america <- ifelse(dat$region == 'NORTH AMERICA', 1, 0)
dat$other_region <- ifelse((dat$region == 'LATIN AMERICA')|
                             (dat$region == 'MIDDLE EAST'), 1, 0 )
dat$region <- NULL

dat$id <- seq(1, nrow(dat))
library(tableone)
library(Matching)

xvars <- c('prog', 'balance', 'honesty', 'pay_plan', 'assignment',
           'forum', 'africa', 'asia', 'europe', 'america', 'other_region')

table1 <- CreateTableOne(vars = xvars, strata = 'tc', data = dat, test = FALSE)
print(table1, smd = TRUE)


table(dat$tc, dat$prog)
num_ctl <- table(dat$tc)[1]
num_trt <- table(dat$tc)[2]
dat[dat$tc == 0, 'prog'] <- rbinom(num_ctl, 1, 0.65)
dat[dat$tc == 1, 'prog'] <- rbinom(num_trt, 1, 0.52)
dat$balance <- ifelse(dat$balance > 0, 1, 0)
dat[is.na(dat1$balance), 'balance'] <- 0
dat[dat$tc == 0, 'balance'] <- rbinom(num_ctl, 1, 0.25)
dat[dat$tc == 0, 'pay_plan'] <- rbinom(num_ctl, 1, 0.52)
dat[dat$tc == 1, 'forum'] <- dat[dat$tc == 1, 'forum'] + rbinom(num_trt, 5, 0.8)
dat[dat$tc == 1, 'asia'] <- rbinom(num_trt, 1, 0.25)
dat[dat$aisa == 1, 'africa'] <- 0
dat[dat$aisa == 1, 'america'] <- 0
dat[dat$aisa == 1, 'europe'] <- 0
dat[dat$aisa == 1, 'other_region'] <- 0

table1 <- CreateTableOne(vars = xvars, strata = 'tc', data = dat, test = FALSE)
print(table1, smd = TRUE)


psmodel <- glm(tc ~ prog + balance + honesty + pay_plan 
              + assignment + forum + africa + asia
              + europe + america + other_region, 
              family = binomial(), data = dat)

dat$ps <- psmodel$fitted.values

logit <- function(p) {log(p)-log(1-p)}
psmatch<-Match(Tr=dat$tc, M=1,X=logit(dat$ps),replace=FALSE,caliper= 1.3)
matched<-dat[unlist(psmatch[c("index.treated","index.control")]), ]

#get standardized differences
matchedtab<-CreateTableOne(vars=xvars, strata ="tc", 
                            data=matched, test = FALSE)
print(matchedtab, smd = TRUE)



plot(dat$ps, jitter(dat$tc+0.5), pch=16, cex=0.5, col=2*dat$tc+2,
     axes=FALSE, type="p", xlim=c(0,1), ylim=c(0,2), xlab="", ylab="")
axis(1)
mtext(c("trt=0", "trt=1"), side=2, at=c(0.5, 1.5), line=3, las=1, adj=0)
mtext("Propensity score", side=1, at=0.5, line=2.5)
lines(quantile(dat$ps[dat$tc==0])[c(2,4)], c(0.85, 0.85), col=2)
lines(rep(quantile(dat$ps[dat$tc==0])[3], 2), c(0.8, 0.9), col=2)
lines(quantile(dat$ps[dat$tc==1])[c(2,4)], c(1.15, 1.15), col=4)
lines(rep(quantile(dat$ps[dat$tc==1])[3], 2), c(1.1, 1.2), col=4)

#################
plot(matched$ps, jitter(matched$tc+0.5), pch=16, cex=0.5, col=2*matched$tc+2,
     axes=FALSE, type="p", xlim=c(0,1), ylim=c(0,2), xlab="", ylab="")
axis(1)
mtext(c("trt=0", "trt=1"), side=2, at=c(0.5, 1.5), line=3, las=1, adj=0)
mtext("Propensity score", side=1, at=0.5, line=2.5)
lines(quantile(matched$ps[matched$tc==0])[c(2,4)], c(0.85, 0.85), col=2)
lines(rep(quantile(matched$ps[matched$tc==0])[3], 2), c(0.8, 0.9), col=2)
lines(quantile(matched$ps[matched$tc==1])[c(2,4)], c(1.15, 1.15), col=4)
lines(rep(quantile(matched$ps[matched$tc==1])[3], 2), c(1.1, 1.2), col=4)


set.seed(12345)


trt_mth_id <- matched[matched$tc == 1, 'id']
ctl_mth_id <- matched[matched$tc == 0, 'id']
trt_unmth_id <- dat[(!dat$id %in% matched$id) & (dat$tc == 1), 'id']
ctl_unmth_id <- dat[(!dat$id %in% matched$id) & (dat$tc == 0), 'id']

dat[dat$id %in% trt_mth_id, 'ret'] <- rbinom(length(trt_mth_id), 1, 0.8)
dat[dat$id %in% ctl_mth_id, 'ret'] <- rbinom(length(ctl_mth_id), 1, 0.77)
dat[dat$id %in% trt_unmth_id, 'ret'] <- rbinom(length(trt_unmth_id), 1, 0.8)
dat[dat$id %in% ctl_unmth_id, 'ret'] <- rbinom(length(ctl_unmth_id), 1, 0.6)

trt_pre <- dat[dat$tc == 1, 'ret']
ctl_pre <- dat[dat$tc == 0, 'ret']
t.test(trt_pre, ctl_pre)

trt_mth <- dat[dat$id %in% trt_mth_id, 'ret']
ctl_mth <- dat[dat$id %in% ctl_mth_id, 'ret']
t.test(trt_mth, ctl_mth)

trt_pre_1 <- sum(trt_pre)
trt_pre_all <- length(trt_pre)
ctl_pre_1 <- sum(ctl_pre)
ctl_pre_all <- length(ctl_pre)

trt_mth_1 <- sum(trt_mth)
trt_mth_all <- length(trt_mth)
ctl_mth_1 <- sum(ctl_mth)
ctl_mth_all <- length(ctl_mth)

prop.test(c(trt_pre_1, ctl_pre_1), c(trt_pre_all, ctl_pre_all), alternative = 'greater')

prop.test(c(trt_mth_1, ctl_mth_1), c(trt_mth_all, ctl_mth_all), alternative = 'greater')

setwd('C:\\Users\\alin\\Documents\\SelfStudy\\CausalEffectsVideos')
saveRDS(dat, file="psm.rds")
dat <- readRDS("psm.rds")
