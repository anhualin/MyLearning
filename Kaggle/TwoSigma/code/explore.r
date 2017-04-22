packages <- c("jsonlite", "dplyr", "purrr", "tidytext", "ggplot2", "lubridate")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)


if (Sys.info()[1] == 'Windows'){
  setwd('C:/Users/alin/Documents/SelfStudy/MyLearning/Kaggle/TwoSigma/data')
}else{
  setwd('/home/alin/MyLearning/Kaggle/TwoSigma/data')  
}


data <- fromJSON("train.json")

# unlist every variable except `photos` and `features` and convert to tibble
vars <- setdiff(names(data), c("photos", "features"))
#library(tibble)?
#library(purrr)
train_df <- map_at(data, vars, unlist) %>% 
            tibble::as_tibble(.) %>%
            mutate(interest_level = factor(interest_level, c("low", "medium", "high")))

pull <- function(x,y) {
    x[,if(is.name(substitute(y))) deparse(substitute(y)) else y, drop = FALSE][[1]]
  }
x <- train_df %>% pull('day')
## seti 
library(syuzhet)
library(DT)

library(tidyr)
library(stringr)
dejunk <- function(a){
  a <- gsub('<a\\s+website_redacted', '',a)
  a <- gsub('<\\S+\\s*/*>', ' ', a)
  a <- gsub('\\S+\\s*@\\s*\\S+', ' ', a)
  a <- gsub('\\d+[-]*\\d+[-]\\d+', ' ', a)
  a <- gsub('\\W+', ' ', a)
  a <- gsub('[[:digit:]]', ' ', a)
}

raw_description <- data[c(9,5)] %>%
  tibble::as_tibble(.) %>%
  mutate(listing_id = unlist(listing_id), description = dejunk(description)) 
  

tidy_train <- raw_description %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words)

word_cnt <- tidy_train %>% 
  count(listing_id) %>%
  mutate(word_cnt = n)


train0a <- train0 %>%
  left_join(word_cnt)

train0a[is.na(train0a$word_cnt), 'word_cnt'] <- 0

senti <- tidy_train %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(listing_id) %>% 
  summarise(sentiment1 = sum(score))

negation_words <- c("not", "no", "never", "without")

neg_senti <- raw_description %>%
  unnest_tokens(bigram, description, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(word1 %in% negation_words) %>% 
  inner_join(get_sentiments("anfinn"), by = c(word2 = "word")) %>%
  group_by(listing_id) %>%
  summarise(sentiment2 = -2*sum(score))
  
sentiment <- senti %>%
  left_join(neg_senti)

sentiment[is.na(sentiment$sentiment2), 'sentiment2'] <- 0
sentiment <- sentiment %>%
  mutate(sentiment = sentiment1 + sentiment2) %>%
  select(listing_id, sentiment)




#### features ####
library(tidyr)
library(stringr)
raw_features <- data[c(9,7)] %>%
  tibble::as_tibble(.) %>%
  mutate(listing_id = unlist(listing_id)) 
  
feature_df <- data[c(9,7)] %>%
  tibble::as_tibble(.) %>%
  mutate(listing_id = unlist(listing_id)) 
  

feature_df$features <- apply(feature_df[,2], 1, function(l) tolower(paste(unlist(l), collapse = '|')))

feature_df <- feature_df %>%
  transform(features = strsplit(features, '|', fixed = TRUE)) %>%
  unnest(features) %>%
  tibble::as_tibble(.) 

pet_df <- feature_df %>%
  filter(str_detect(features, '(^|\\s)(pet|cat|dog)(s$|s\\s|\\s|$)' )) %>%
  mutate(pet = 1 - 2* as.integer(str_detect(features, '(^|\\s)no\\s+(pet|cat|dog)'))) %>%
  select(listing_id, pet) %>%
  group_by(listing_id) %>% 
  summarise(pet = max(pet))

laundry_df <- feature_df %>%
  filter(str_detect(features, 'laundry')) %>%
  mutate(laundry = 1) %>%
  select(listing_id, laundry) %>%
  group_by(listing_id) %>%
  summarise(laundry = max(laundry))

fee_df <- feature_df %>%
  filter(str_detect(features, '(\\W)fee(\\W|$)')) %>%
  mutate(fee = 2 * as.integer(str_detect(features, '(^|\\s)no(\\W)')) - 1) %>%
  select(listing_id, fee) %>%
  group_by(listing_id) %>%
  summarise(fee = max(fee))


pool_df <- feature_df %>%
  filter(str_detect(features, '(\\W|^)pool(\\W|$)')) %>%
  mutate(pool = 1) %>%
  select(listing_id, pool) %>%
  group_by(listing_id) %>%
  summarise(pool = max(pool))






feature1_df <- raw_features %>%
  left_join(pet_df) %>%
  select(listing_id, pet)
feature1_df[is.na(feature1_df$pet), 'pet'] <- 0

df <- data_frame(id = c(1,2,3,4,5), a = c(1,2,3,4,5))
df1 <- data_frame(id = c(1,2,3), b = c(1,2,3))

x <-df %>%
  left_join(df1)
x[is.na(x$b), 'b'] <- 0

x$b1 <- apply(x, 1, function(g) if (is.na(g)) 0 else g )
x <- x %>%
  mutate(b = if (is.na(b)) 0 else b)

df <- data_frame(a = c('pet friendly', 'pets ok', ' pet ok', 'pettion', 
                       'no pet', 'apeta', 'apet', 'petsg',
                       'cat friendly', 'dogs ok', ' cat ok', 'lcatg do',
                       ' no dog', 'ldog', 'acat', 'lcatdog'))

df %>%
  filter(str_detect(a, '(^|\\s)(pet|cat|dog)(s$|s\\s|\\s|$)' )) 
         
         | 

df %>%
  filter(str_detect(a, 'pet$'))


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
