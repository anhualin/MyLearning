require(ggplot2)
require(ROCR)
require(dplyr)

data <- read.csv(file = 'C:\\Users\\alin\\Documents\\Data\\FDA2018\\Collection_Risk.csv')

names(data) <- tolower(names(data))

data$id <- seq(1, nrow(data))

data$student_id <- NULL

outcome <- 'risk'
pos <- '1'

data$risk_n <- data$risk
data$risk <- as.factor(data$risk)

train <- data[data$ind <= 10, ]
test <- data[data$ind == 11, ]

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

single_cat(train, 'college')

ggplot(data=train) +
  geom_density(aes(x = bb_activity_latest,
                   color = risk,
                   linetype = risk ))

summary(train$bb_gip_etc_ratio_std)

ggplot(data=train) +
  geom_density(aes(x = cum_gpa,
                   color = risk,
                   linetype = risk ))

vars <- setdiff(names(train), c('student_id', 'risk','bb_gip_etc_ratio_std',
                                'cp_prior3mon_pay_cnt', 'payment_auto_flag',
                                'payment_auto_flag', 'ind',
                                'nation_desc', 'id', 'risk_n'))

catVars <- vars[sapply(train[,vars],class) %in%
                  c('factor','character')]
numericVars <- vars[sapply(train[,vars],class) %in%
                      c('numeric','integer')]

train0 <- train[train$ind <= 9, ]
valid0 <- train[train$ind == 10, ]

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
  cuts[[1]] <- cuts[[1]] - 0.1
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}


calcAUC <- function(predcol,outcol) {
  #calculated auc
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

train0_num_sd <- scale(train0[, numericVars])
means <- attr(train0_num_sd, "scaled:center")
stds <- attr(train0_num_sd, "scaled:scale")
train0[, numericVars] <- data.frame(train0_num_sd)

# use the same mean/std to transform valid0
valid0_num <- valid0[, numericVars]
valid0_num <- t(apply(valid0_num, 1, '-', means))
valid0_num <- t(apply(valid0_num, 1, '/', stds))
valid0[, numericVars] <- data.frame(valid0_num)


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


v <- 'program'
pi <- paste('pred',v,sep='')
# train0[,pi] <- mkPredC(train0[,outcome],train0[,v],train0[,v])
# valid0[,pi] <- mkPredC(train0[,outcome],train0[,v],valid0[,v])
# aucTrain <- calcAUC(train0[,pi], train0[,outcome])
# aucCal <- calcAUC(valid0[,pi], valid0[,outcome])

eval <- prediction(train0[,pi], train0[, outcome])
plot(performance(eval,"tpr","fpr"))
print(attributes(performance(eval,'auc'))$y.values[[1]])

eval <- prediction(valid0[,pi], valid0[, outcome])
plot(performance(eval,"tpr","fpr"))
print(attributes(performance(eval,'auc'))$y.values[[1]])

v <- 'ba_credits_passed_prior1yr'
pi <- paste('pred',v,sep='')
# train0[,pi] <- mkPredC(train0[,outcome],train0[,v],train0[,v])
# valid0[,pi] <- mkPredC(train0[,outcome],train0[,v],valid0[,v])
# aucTrain <- calcAUC(train0[,pi], train0[,outcome])
# aucCal <- calcAUC(valid0[,pi], valid0[,outcome])
eval <- prediction(valid0[,pi], valid0[, outcome])
plot(performance(eval,"tpr","fpr"))
print(attributes(performance(eval,'auc'))$y.values[[1]])




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
varScore