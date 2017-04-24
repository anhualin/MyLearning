if (Sys.info()[1] == 'Windows'){
  setwd('C:/Users/alin/Documents/SelfStudy/MyLearning/Kaggle/TwoSigma/data')
}else{
  setwd('/home/alin/MyLearning/Kaggle/TwoSigma/data')  
}

library(data.table)
library(jsonlite)

library(lubridate)
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)


# Load data
t1 <- fromJSON("train.json")
vars <- setdiff(names(t1), c("photos", "features"))
t1 <- map_at(t1, vars, unlist) %>% tibble::as_tibble(.)

library(caTools)

split <- sample.split((1:nrow(t1)), SplitRatio = 0.7)
train_df <- t1[split,]
valid_df <- t1[!split, ]

t1 <- train_df
s1 <- valid_df

# Rate By Level using Manager ID
row_by_manager_id = aggregate(rep(1, nrow(t1)),by=list(t1$manager_id), sum)
names(row_by_manager_id) <- c("manager_id","count")
row_by_manager_id$freq = row_by_manager_id$count/49352
row_by_manager_id$manager_odds = log(row_by_manager_id$freq/(1-row_by_manager_id$freq))
t1 = merge(t1,row_by_manager_id,by="manager_id")
t1 <- subset(t1,select = -c(count,freq))

# Rate By Level using Building ID
row_by_building_id = aggregate(rep(1, nrow(t1)),by=list(t1$building_id), sum)
names(row_by_building_id) <- c("building_id","count")
row_by_building_id$freq = row_by_building_id$count/49352
row_by_building_id$building_odds = log(row_by_building_id$freq/(1-row_by_building_id$freq))
t1 = merge(t1,row_by_building_id,by="building_id")
t1 <- subset(t1, select = -c(count,freq))

# Rate By Level using Listing ID
row_by_listing_id = aggregate(rep(1, nrow(t1)),by=list(t1$listing_id), sum)
names(row_by_listing_id) <- c("listing_id","count")
row_by_listing_id$freq = row_by_listing_id$count/49352
row_by_listing_id$listing_odds = log(row_by_listing_id$freq/(1-row_by_listing_id$freq))
t1 = merge(t1,row_by_listing_id,by="listing_id")
t1 <- subset(t1, select = -c(count,freq))

t2 <- data.table(bathrooms=unlist(t1$bathrooms)
                 ,bedrooms=unlist(t1$bedrooms)
                 ,building_id=as.factor(unlist(t1$building_id))
                 ,building_odds=as.numeric(unlist(t1$building_odds))
                 ,created=as.POSIXct(unlist(t1$created))
                 # ,description=unlist(t1$description) # parse errors
                 # ,display_address=unlist(t1$display_address) # parse errors
                 ,num_features=as.numeric(lengths(t1$features))
                 ,num_photos=as.numeric(lengths(t1$photos))
                 ,latitude=unlist(t1$latitude)
                 ,longitude=unlist(t1$longitude)
                 ,listing_id=unlist(t1$listing_id)
                 ,listing_odds=as.numeric(t1$listing_odds)
                 ,manager_id=as.factor(unlist(t1$manager_id))
                 ,manager_odds=as.numeric(t1$manager_odds)
                 ,price=unlist(t1$price)
                 ,logprice=as.numeric(log(t1$price))
                 ,interest_level=as.factor(unlist(t1$interest_level))
                 # ,street_adress=unlist(t1$street_address) # parse errors
)
t2[,":="(yday=yday(created)
         ,month=month(created)
         ,mday=mday(created)
         ,wday=wday(created)
         ,hour=hour(created))]



vars <- setdiff(names(s1), c("photos", "features"))
s1 <- map_at(s1, vars, unlist) %>% tibble::as_tibble(.)
# Rate By Level using Manager ID
row_by_manager_id = aggregate(rep(1, nrow(s1)),by=list(s1$manager_id), sum)
names(row_by_manager_id) <- c("manager_id","count")
row_by_manager_id$freq = row_by_manager_id$count/49352
row_by_manager_id$manager_odds = log(row_by_manager_id$freq/(1-row_by_manager_id$freq))
s1 = merge(s1,row_by_manager_id,by="manager_id")
s1 <- subset(s1, select = -c(count,freq))

# Rate By Level using Building ID
row_by_building_id = aggregate(rep(1, nrow(s1)),by=list(s1$building_id), sum)
names(row_by_building_id) <- c("building_id","count")
row_by_building_id$freq = row_by_building_id$count/49352
row_by_building_id$building_odds = log(row_by_building_id$freq/(1-row_by_building_id$freq))
s1 = merge(s1,row_by_building_id,by="building_id")
s1 <- subset(s1, select = -c(count,freq))

# Rate By Level using Listing ID
row_by_listing_id = aggregate(rep(1, nrow(s1)),by=list(s1$listing_id), sum)
names(row_by_listing_id) <- c("listing_id","count")
row_by_listing_id$freq = row_by_listing_id$count/49352
row_by_listing_id$listing_odds = log(row_by_listing_id$freq/(1-row_by_listing_id$freq))
s1 = merge(s1,row_by_listing_id,by="listing_id")
s1 <- subset(s1, select = -c(count,freq))

s2 <- data.table(bathrooms=unlist(s1$bathrooms)
                 ,bedrooms=unlist(s1$bedrooms)
                 ,building_id=as.factor(unlist(s1$building_id))
                 ,building_odds=as.numeric(unlist(s1$building_odds))
                 ,created=as.factor(unlist(s1$created))
                 # ,description=unlist(s1$description) # parse errors
                 # ,display_address=unlist(s1$display_address) # parse errors
                 ,num_features=as.numeric(lengths(s1$features))
                 ,num_photos=as.numeric(lengths(s1$photos))
                 ,latitude=unlist(s1$latitude)
                 ,longitude=unlist(s1$longitude)
                 ,listing_id=unlist(s1$listing_id)
                 ,listing_odds=unlist(s1$listing_odds)
                 ,manager_id=as.factor(unlist(s1$manager_id))
                 ,manager_odds=as.numeric(unlist(s1$manager_odds))
                 ,price=unlist(s1$price)
                 ,logprice=as.numeric(log(s1$price))
                 # ,street_adress=unlist(s1$street_address) # parse errors
)
s2[,":="(yday=yday(created)
         ,month=month(created)
         ,mday=mday(created)
         ,wday=wday(created)
         ,hour=hour(created))]


library(h2o)
h2o.init(nthreads = -1, max_mem_size="10g")

train <- as.h2o(t2[,-"created"], destination_frame = "train.hex")

varnames <- setdiff(colnames(train), "interest_level")
notUse <- c('yday', 'month', 'mday', 'wday', 'hour', 'manager_odds', 'building_odds')
varnames <- setdiff(varnames, notUse)

gbm1 <- h2o.gbm(x = varnames, y = "interest_level", train, seed = 10000)

gbm1 <- h2o.gbm(x = varnames
                ,y = "interest_level"
                ,training_frame = train
                ,distribution = "multinomial"
                ,model_id = "gbm1"
                #,nfolds = 5
                ,ntrees = 750
                ,learn_rate = 0.05
                ,max_depth = 7
                ,min_rows = 20
                ,sample_rate = 0.7
                ,col_sample_rate = 0.7
                #   ,stopping_rounds = 5
                #   ,stopping_metric = "logloss"
                #   ,stopping_tolerance = 0
                ,seed=321
)
test <- as.h2o(s2[,-"created"], destination_frame = "test.hex")

predH2o <- as.data.frame(h2o.predict(gbm1, test))
test_result1 <- cbind(s1, predH2o)
test_result <- test_result1 %>%
  select(listing_id, interest_level, high, medium, low)

logloss <- function(result){
  score <- 0
  for(i in 1:nrow(result)){
    score <- score - log(result[i, as.character(result[i, 'interest_level'])])
  }  
  score <- score/nrow(result)
} 

print(logloss(test_result))
