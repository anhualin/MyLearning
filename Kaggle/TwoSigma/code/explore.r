packages <- c("jsonlite", "dplyr", "purrr", "tidytext", "ggplot2")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

setwd('C:/Users/alin/Documents/Kaggle/TwoSigma/data')
data <- fromJSON("train.json")

# unlist every variable except `photos` and `features` and convert to tibble
vars <- setdiff(names(data), c("photos", "features"))
library(tibble)
library(purrr)
train_df <- map_at(data, vars, unlist) %>% 
            tibble::as_tibble(.) %>%
            mutate(interest_level = factor(interest_level, c("low", "medium", "high")))

ggplot(data=train_df) +
  geom_density(aes(x = bathrooms,
                   color = interest_level,
                   linetype = interest_level ))

pull <- function(x,y) {
    x[,if(is.name(substitute(y))) deparse(substitute(y)) else y, drop = FALSE][[1]]
  }
x <- train_df %>% pull('bedrooms')
## seti anal
library(syuzhet)
library(DT)
sentiment <- get_nrc_sentiment(train_df$description)
datatable(head(sentiment))

train_set <- cbind(train_df, sentiment)

sent1 <- get_sentiment(s_v, method="syuzhet")

# basic tidy text
data <- map_at(data, vars, unlist) %>% 
  tibble::as_tibble(.) %>%
  select(listing_id, features, interest_level) %>%
  mutate(interest_level = factor(interest_level, c("low", "medium", "high")))

head(data, n =5)
library(tidry)
tidy_data <- data %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features) %>%
  unnest_tokens(word, features)

head(tidy_data, n = 5)

## h2o ###
library(data.table)
library(jsonlite)
library(h2o)
library(lubridate)
h2o.init(nthreads = -1, max_mem_size="8g")
# Load data
t1 <- fromJSON("train.json")
# There has to be a better way to do this while getting repeated rows for the "feature" and "photos" columns
t2 <- data.table(bathrooms=unlist(t1$bathrooms)
                 ,bedrooms=unlist(t1$bedrooms)
                 #,building_id=as.factor(unlist(t1$building_id))
                 #,created=as.POSIXct(unlist(t1$created))
                 # ,description=unlist(t1$description) # parse errors
                 # ,display_address=unlist(t1$display_address) # parse errors
                 ,latitude=unlist(t1$latitude)
                 ,longitude=unlist(t1$longitude)
                 #,listing_id=unlist(t1$listing_id)
                 #,manager_id=as.factor(unlist(t1$manager_id))
                 ,price=unlist(t1$price)
                 ,interest_level=as.factor(unlist(t1$interest_level))
                 # ,street_adress=unlist(t1$street_address) # parse errors
)
# t2[,":="(yday=yday(created)
#          ,month=month(created)
#          ,mday=mday(created)
#          ,wday=wday(created)
#          ,hour=hour(created))]

library(caTools)
split <- sample.split((1:nrow(t2)), SplitRatio = 0.7)
train <- t2[split]
test <- t2[!split]

varnames <- setdiff(names(train), "listing_id")
features <- setdiff(varnames, 'interest_level')
train1 <- as.data.frame(train)
train.hex <- as.h2o(train1)

# varnames <- setdiff(colnames(train), "interest_level")
h2o_gbm <- h2o.gbm(x = features, y = "interest_level", train.hex, seed = 10000)
h2o_gbm <- h2o.gbm(1:5, 6, train.hex, seed = 10000)

true_interest <- test$interest_level
test1 <- as.data.frame(test)
test1$interest_level <- NULL
test.hex <- as.h2o(test1)
predH2o <- as.data.frame(h2o.predict(h2o_gbm, test.hex))
pred <- predH2o$predict
table(true_interest, pred)

# gbm1 <- h2o.gbm(x = features
#                 ,y = "interest_level"
#                 ,training_frame = train
#                 ,distribution = "multinomial"
#                 ,model_id = "gbm1"
#                 #,nfolds = 5
#                 ,ntrees = 750
#                 ,learn_rate = 0.05
#                 ,max_depth = 7
#                 ,min_rows = 20
#                 ,sample_rate = 0.7
#                 ,col_sample_rate = 0.7
#                 #   ,stopping_rounds = 5
#                 #   ,stopping_metric = "logloss"
#                 #   ,stopping_tolerance = 0
#                 ,seed=321
# )

# EDA
summary(train_df$interest_level)
