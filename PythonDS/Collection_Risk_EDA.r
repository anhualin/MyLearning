require(ggplot2)
require(ROCR)
require(dplyr)


data <- read.table('Collection_Risk.csv', header = TRUE, sep = ',')
names(data) <- tolower(names(data))

outcome <- 'risk'
pos <- '1'


num_to_cat <- c('risk', 'in_module', 'in_future', 'in_dissertation')
for (fac in num_to_cat){
  data[, fac] <- as.factor(data[, fac])
}

#utility functions to explore single feature v.s. the target
single_cat <- function(dSet, feature, target = 'risk') {
  #for categorical feature
  print(sum(is.na(dSet[, feature])))
  tab <- table(feature = as.factor(dSet[, feature]), 
                                   target = dSet[, target])
  print(tab)
  print(tab[,2]/(tab[, 1] + tab[, 2] ))
}


single_num <- function(dSet, feature, target = 'risk'){
  #for numerical feature
  print(sum(is.na(dSet[, feature])))
  print(summary(dSet[, feature]))
  plot(dSet[,feature] ~ dSet[, target])
  print(summary(dSet[dSet[, target]  != pos, feature]))
  print(summary(dSet[dSet[, target]  == pos, feature]))
}  


# use July, August and Sept for training, Oct for testing
train <- data[data$ind <= 9, ]
test <- data[data$ind == 10, ]

#0 target
table(train[, outcome]) 
#reasonable balanced


#1 future
single_num(train, 'future')
ggplot(train) + geom_density(aes(x=future))
#no NA.
#nearly 50% < 10, also significant around 1100
# and it seems to suggest that people with certain future balance
# are more likely to pay? makes sense?

#2 aging
# step increase by 30 days
single_num(train, 'aging')
single_cat(train, 'aging')
# no NA.
# clearly longer aging --> higher risk

#3 program
single_cat(train, 'program')
#no NA
# the risk somewhat varies by programs,

#4 degree
single_cat(train, 'degree')
#no NA
# no significant difference across degrees, 
# MA has  somewhat less risk rate

#5 college
single_cat(train, 'college')
# no NA
# There is some observable difference across colleges

#6 bb_activity_latest
single_num(train, 'bb_activity_latest')
ggplot(data=train) +
  geom_density(aes(x = bb_activity_latest,
                   color = risk,
                   linetype = risk ))

# agrees with intuition, people with recent activities are less risky
# could be a good factor


#7 bb_gip_etc_ratio_std
single_num(train, 'bb_gip_etc_ratio_std')
# 1975 missing value, quite a lot, nearly 25%,
# will not use unless really necessary

#8 cum_gpa
single_num(train, 'cum_gpa')
ggplot(data=train) +
  geom_density(aes(x = cum_gpa,
                   color = risk,
                   linetype = risk ))
#no NA
# risky students generally have lower gpa

#9 cum_credits
single_num(train, 'cum_credits')
#no NA, agrees with intuition, more credits --> less risky

#10. ba_credits_passed_prior1yr
single_num(train, 'ba_credits_passed_prior1yr')
ggplot(data=train) +
  geom_density(aes(x = ba_credits_passed_prior1yr,
                   color = risk,
                   linetype = risk ))

#no NA, reasonable range
# more credits --> less risky


#11 ba_age
single_num(train, 'ba_age')
summary(train$ba_age)
# 3 NA's,  some wrong data: age = 121, or 0?
# seems not very useful, will not consider in 1st round

#12 gender
single_cat(train, 'gender')
# 3 blank, 16 Unknown
# set blank to unknown
train[train$gender == '', 'gender'] = 'N'
train$gender <- droplevels(train$gender)
# very few N, keeping them in case N appears in the test data
# might be marginally useful, but case N has too few instances.

#13 cp_prior3mon_pay_cnt
# directly derived from cp_prior3mon_pay_total
single_num(train, 'cp_prior3mon_pay_cnt')
single_cat(train, 'cp_prior3mon_pay_cnt')
# it seems that the big difference is whether it's 0 or >0,
# probably cp_prior3mon_pay_total is sufficient


#14 cp_prior3mon_pay_total
single_num(train, 'cp_prior3mon_pay_total')
# 1 negative -814.8, error? will remove this row 
# 1 big positive > 2700 = 6717, outlier? will remove this row
train <- train[train$cp_prior3mon_pay_total >= 0 &
                 train$cp_prior3mon_pay_total < 2700, ]
single_num(train, 'cp_prior3mon_pay_total')
ggplot(data=train) +
  geom_density(aes(x = cp_prior3mon_pay_total,
                   color = risk,
                   linetype = risk ))
# more payment --> less risky

#15 payment_auto_flag
# useless

#16 risk


#17 nation_desc
# 160 levels with many low count,
# will not consider in 1st round
a <- levels(train$nation_desc)
lowNation <- c('')
for(i in 1:length(a)){
  if (sum(train$nation_desc == a[i]) <= 50){
    lowNation <- c(lowNation, a[i])
  }
}

train$nation_tr <- as.character(train$nation_desc)
train[train$nation_tr %in% lowNation, ]$nation_tr <- 'Other'
train$nation_tr <- as.factor(train$nation_tr)
train$nation_tr <- droplevels(train$nation_tr)

single_cat(train, 'nation_tr')


#18 current_balance
single_num(train, 'current_balance')
ggplot(data=train) +
  geom_density(aes(x = current_balance,
                   color = risk,
                   linetype = risk ))
# less balance -> less risky

#19 module_cp
single_num(train, 'module_cp')
# no NA, generally more module, less risky

#20 in_module
single_cat(train, 'in_module')
# people in module are significantly less risky

#21 in_future
single_cat(train, 'in_future')
# people who have registered a future module
# are significantly less risky

#22 in_dissertation
single_cat(train, 'in_dissertation')
# people who are in a dissertation module are less risky,
# but few people are in such modules


#23 day_since_end
single_num(train, 'day_since_end')
ggplot(data=train) +
  geom_density(aes(x = day_since_end,
                   color = risk,
                   linetype = risk ))
# no clear significant difference

#24 day_since_begin
single_num(train, 'day_since_begin')
# correlated with in_module

#25 day_to_begin
single_num(train, 'day_to_begin')
# likely correlated with in_future


#####################################################
vars <- setdiff(names(train), c('student_id', 'risk','bb_gip_etc_ratio_std',
                                    'cp_prior3mon_pay_cnt', 'payment_auto_flag',
                                    'payment_auto_flag', 'nation_desc', 'ind'))

catVars <- vars[sapply(train[,vars],class) %in%
                  c('factor','character')]
numericVars <- vars[sapply(train[,vars],class) %in%
                      c('numeric','integer')]

# some variables are highly correlated,
# for example, module_cp, cum_gpa, cum_credits etc.

#########################################################3
# explore one-variable model
mkPredC <- function(outCol,varCol,appCol) {
  pPos <- sum(outCol==pos)/length(outCol)
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(pred)] <- pPos
  pred
}


mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,
                                     probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}


calcAUC <- function(predcol,outcol) {
  #calculated auc
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

# use July and August as train, Sep as validation 
train0 <- train[train$ind <= 8, ]
valid0 <- train[train$ind == 9, ]

#scale and centralize numerical variables of train0 
train0_num_sd <- scale(train0[, numericVars])
means <- attr(train0_num_sd, "scaled:center")
stds <- attr(train0_num_sd, "scaled:scale")
train0[, numericVars] <- data.frame(train0_num_sd)

# use the same mean/std to transform valid0
valid0_num <- valid0[, numericVars]
valid0_num <- t(apply(valid0_num, 1, '-', means))
valid0_num <- t(apply(valid0_num, 1, '/', stds))
valid0[, numericVars] <- data.frame(valid0_num)

### check auc for one-variable models
for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  train0[,pi] <- mkPredN(train0[,outcome],train0[,v],train0[,v])
  valid0[,pi] <- mkPredN(train0[,outcome],train0[,v],valid0[,v])
  aucTrain <- calcAUC(train0[,pi],train0[,outcome])
  aucCal <- calcAUC(valid0[,pi], valid0[,outcome])
  print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                pi,aucTrain,aucCal))
}

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  train0[,pi] <- mkPredC(train0[,outcome],train0[,v],train0[,v])
  valid0[,pi] <- mkPredC(train0[,outcome],train0[,v],valid0[,v])
  aucTrain <- calcAUC(train0[,pi], train0[,outcome])
  aucCal <- calcAUC(valid0[,pi], valid0[,outcome])
  print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                pi,aucTrain,aucCal))
  
}


# some potential good features:
# bb_activity_latest, ba_credits_passed_prior1yr,
# cp_prior3mon_pay_total, module_cp,
# in_module, in_future


# Variable ordering by AIC-variant using train0/valid0
logLikelyhood <- function(outCol,predCol) {
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}


selVars <- c()
scores <- c()
baseRateCheck <- logLikelyhood(
  valid0[,outcome],
  sum(valid0[,outcome]==pos)/length(valid0[,outcome]))
print(baseRateCheck)
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(valid0[,outcome],valid0[,pi]) -
                   baseRateCheck))
  
  print(sprintf("%s, calibrationScore: %g", pi,liCheck))
  selVars <- c(selVars,v)
  scores <- c(scores, liCheck)
  
}

for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(valid0[,outcome],valid0[,pi]) -
                   baseRateCheck) - 1)
  
  print(sprintf("%s, calibrationScore: %g", pi,liCheck))
  selVars <- c(selVars,v)
  scores <- c(scores, liCheck)
  
}

varScore <- data.frame(var = selVars, score = scores)
varScore <- varScore[with(varScore, order(-score)),]

############################################################
# try several models with default setting to select variables
getRate <- function(predcol, outcol, num_decile = 10){
  #utility function to get risk rate of each decile
  tmpd <- data.frame(risk = predcol, outcome = outcol)
  tmpd$decile <- num_decile + 1 - ntile(tmpd$risk, num_decile)
  decile_rate <- rep(0, num_decile)
  for(i in 1:num_decile){
    decile <- tmpd[tmpd$decile == i, ]
    decile_rate[i] <- sum(decile$outcome == 1)/nrow(decile)
  }
  decile_rate
}

accuracyMeasures <- function(pred, truth, name="model") {
  # calculated various measures
  ctable <- table(truth=truth,
                  pred=(pred>0.5))
  accuracy <- sum(diag(ctable))/sum(ctable)
  precision <- ctable[2,2]/sum(ctable[,2])
  recall <- ctable[2,2]/sum(ctable[2,])
  f1 <- 2*precision*recall/(precision + recall)
  sensitivity <- recall
  specificity <- ctable[1,1]/sum(ctable[1,])
  decile <- getRate(pred,  truth)
  data.frame(model=name, accuracy=accuracy,
             sensitivity = sensitivity, specificity = specificity,
             precision = precision, f1=f1, decile = decile)
  
  
}


#1. logistic regression
y <- outcome
features <- as.character((varScore$var))
logit_result <- list()


x <- features[1:4]
for(i in 1:15){
  x <- features[1:i]
  fmla <- paste(y, paste(x, collapse = '+'), sep = '~')
  model <- glm(fmla, data = train0, family=binomial(link="logit"))
  pred <- predict(model, newdata=valid0, type="response")
  result <- accuracyMeasures(pred, valid0$risk, name = 'logit')
  logit_result[[i]] <- result
}

# top 4 features are chosen.

# 2 random forest
require(randomForest)
rf_result <- list()
rf_result[[1]] <- "Nothing"


for(i in 2:10){
  x <- features[1:i]
  model <- randomForest(x=train0[, x], y = train0[, outcome])
  pred <- predict(model, newdata=valid0[, x], type='prob')[, pos]
  result <- accuracyMeasures(pred, valid0$risk, name = 'rf')
  rf_result[[i]] <- result
}

# again, the top 4

# gbm
require(h2o)
localH2O = h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # Clean slate - just in case the cluster was already running


x <- features[1:10]
trainh2o <- train0[, c(x,y)]
validh2o <- valid0[,  x]

train.hex <- as.h2o(trainh2o)
train.hex[, outcome] <- as.numeric(train.hex[, outcome])
valid.hex <- as.h2o(validh2o)

gbm_result <- list()
lastVar <- length(trainh2o)
for(i in 1:10){
  h2o_gbm <- h2o.gbm(1:i, lastVar, train.hex, seed = 10000)
  predH2o <- as.data.frame(h2o.predict(h2o_gbm, valid.hex))
  pred <- predH2o$predict
  result <- accuracyMeasures(pred, valid0$risk, name = 'gbm')
  gbm_result[[i]] <- result
}

# So will use the top 4 variables:



# final decision:
# use variables:
# in_future, bb_activity_latest, cp_prior3mon_pay_total, and in_module
# use model: logistic regression


# scale test 
test_num <- test[, numericVars]
test_num <- t(apply(test_num, 1, '-', means))
test_num <- t(apply(test_num, 1, '/', stds))
test[, numericVars] <- data.frame(test_num)



# output for SilverBullet
x <- features[1:4]
usedTrain <- train0[, c(x,y)]
usedTest <- test[, x]
usedTrain$target <- ifelse(usedTrain$risk == pos, "yes", 'no')
#usedTest$target <- ifelse(usedTest$risk == pos, "yes", 'no')
usedTrain$risk <- NULL
#usedTest$risk <- NULL


write.csv(usedTrain, file = "Collection_risk_train.csv", row.names = FALSE)
write.csv(usedTest, file = "Collection_risk_test.csv", row.names = FALSE)

testLabel <- test[, y]
write.csv(testLabel, file = "Collection_risk_test_label.csv", row.names = FALSE)


## for ppt

x <- features[1:4]
fmla <- paste(y, paste(x, collapse = '+'), sep = '~')
model <- glm(fmla, data = train0, family=binomial(link="logit"))
pred <- predict(model, newdata=test, type="response")
result <- accuracyMeasures(pred, test$risk, name = 'logit')
print(result)

test1 <- test
test1$score <- pred
test1$decile <- 11 - ntile(pred, 10)
out <- data.frame(decile = 1:10, num_stu = 1:10, risk_rate = 1:10)
stu <- table(test1$decile)
for(i in 1:10){
  decile <- test1[test1$decile == i, ]
  out[i, 'risk_rate'] = sum(decile[, outcome] == pos)/nrow(decile)
  out[i, 'num_stu'] = stu[i]
}

write.csv(out, file = "Sample_output.csv", row.names = FALSE)

silver1 <- read.table('Collection_risk_out.csv', header = TRUE, sep = ',')
rsilver1 <- accuracyMeasures(silver1$prob1, test$risk)
silver2 <- read.table('Collection_risk_rfout.csv', header = FALSE, sep = ',')
rsilver2 <- accuracyMeasures(silver2$V3, test$risk)


i = 4

  x <- features[1:i]
  model <- randomForest(x=train0[, x], y = train0[, outcome])
  pred <- predict(model, newdata=test[, x], type='prob')[, pos]
  result <- accuracyMeasures(pred, test$risk, name = 'rf')
  rf_result[[i]] <- result
}

