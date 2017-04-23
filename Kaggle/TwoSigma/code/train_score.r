packages <- c("jsonlite", "dplyr", "purrr", "tidytext", "ggplot2", "lubridate", "tidyr",
              "stringr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)


if (Sys.info()[1] == 'Windows'){
  setwd('C:/Users/alin/Documents/SelfStudy/MyLearning/Kaggle/TwoSigma/data')
}else{
  setwd('/home/alin/MyLearning/Kaggle/TwoSigma/data')  
}


raw_train <- fromJSON("train.json")
raw_test <- fromJSON("test.json")

vars <- setdiff(names(raw_train), c("photos", "features"))
train_df <- map_at(data, vars, unlist) %>% 
  tibble::as_tibble(.) %>%
  mutate(interest_level = factor(interest_level, c("low", "medium", "high"))) %>%
  select(-c(building_id, display_address, manager_id, photos, street_address)) %>%
  mutate(cat = 'train')

test_df <- map_at(raw_test, vars, unlist) %>% 
  tibble::as_tibble(.) %>%
  select(-c(building_id, display_address, manager_id, photos, street_address)) %>%
  mutate(cat = 'test', interest_level = NA)

data_df <- rbind(train_df, test_df)

data_df$year <- year(data_df$created)
data_df$month <- month(data_df$created)
data_df$day <- day(data_df$created)

day_by_10 <- function(day){
  floor((min(30, day) - 1) / 10)
}
data_df$ten_day <- unlist(map(data_df$day, day_by_10))

dejunk <- function(a){
  a <- gsub('<a\\s+website_redacted', '',a)
  a <- gsub('<\\S+\\s*/*>', ' ', a)
  a <- gsub('\\S+\\s*@\\s*\\S+', ' ', a)
  a <- gsub('\\d+[-]*\\d+[-]\\d+', ' ', a)
  a <- gsub('\\W+', ' ', a)
  a <- gsub('[[:digit:]]', ' ', a)
}

raw_description <- data_df %>%
  select(listing_id, description) %>%
  mutate(description = dejunk(description)) 

tidy_description <- raw_description %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words)

word_cnt <- tidy_description %>% 
  count(listing_id) %>%
  mutate(word_cnt = n) %>%
  select(listing_id, word_cnt)

senti <- tidy_description %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(listing_id) %>% 
  summarise(sentiment1 = sum(score))

negation_words <- c("not", "no", "never", "without")

neg_senti <- raw_description %>%
  unnest_tokens(bigram, description, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(word1 %in% negation_words) %>% 
  inner_join(get_sentiments("afinn"), by = c(word2 = "word")) %>%
  group_by(listing_id) %>%
  summarise(sentiment2 = -2*sum(score))

sentiment <- senti %>%
  left_join(neg_senti)

sentiment[is.na(sentiment$sentiment2), 'sentiment2'] <- 0
sentiment <- sentiment %>%
  mutate(sentiment = sentiment1 + sentiment2) %>%
  select(listing_id, sentiment)

raw_features <- data_df %>%
  select(listing_id, features)

raw_features$features <- apply(raw_features[,2], 1, function(l) tolower(paste(unlist(l), collapse = '|')))

feature_df <- raw_features %>%
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

pool_df <- feature_df %>%
  filter(str_detect(features, '(\\W|^)pool(\\W|$)')) %>%
  mutate(pool = 1) %>%
  select(listing_id, pool) %>%
  group_by(listing_id) %>%
  summarise(pool = max(pool))

fee_df <- feature_df %>%
  filter(str_detect(features, '(\\W)fee(\\W|$)')) %>%
  mutate(fee = 2 * as.integer(str_detect(features, '(^|\\s)no(\\W)')) - 1) %>%
  select(listing_id, fee) %>%
  group_by(listing_id) %>%
  summarise(fee = max(fee))

dishwasher_df <- feature_df %>%
  filter(str_detect(features, '(\\W|^)dishwasher(\\W|$)')) %>%
  mutate(dishwasher = 1) %>%
  select(listing_id, dishwasher) %>%
  group_by(listing_id) %>%
  summarise(dishwasher = max(dishwasher))

final_df <- data_df %>%
  left_join(word_cnt) %>%
  left_join(sentiment) %>%
  left_join(pet_df) %>%
  left_join(laundry_df) %>%
  left_join(pool_df) %>%
  left_join(fee_df) %>%
  left_join(dishwasher_df)

final_df[is.na(final_df$word_cnt), 'word_cnt'] <- 0
final_df[is.na(final_df$sentiment), 'sentiment'] <- 0
final_df[is.na(final_df$pet), 'pet'] <- 0
final_df[is.na(final_df$laundry), 'laundry'] <- 0
final_df[is.na(final_df$pool), 'pool'] <- 0
final_df[is.na(final_df$fee), 'fee'] <- 0
final_df[is.na(final_df$dishwasher), 'dishwasher'] <- 0

final_df$word_cnt <- as.factor(final_df$word_cnt)
final_df$sentiment <- as.factor(final_df$sentiment)
final_df$pet <- as.factor(final_df$pet)
final_df$laundry <- as.factor(final_df$laundry)
final_df$pool <- as.factor(final_df$pool)
final_df$fee <- as.factor(final_df$fee)
final_df$dishwasher <- as.factor(final_df$dishwasher)

final_df <- final_df %>%
  select(cat, listing_id, bathrooms, bedrooms, latitude, longitude,
         price, month, ten_day, word_cnt, sentiment, pet, laundry,
         pool, fee, dishwasher, interest_level) %>%
  filter(cat == 'test' | price < 50000)

library(h2o)
h2o.init(nthreads = -1, max_mem_size="10g")

train_df <- final_df %>%
  filter(cat == 'train')
library(caTools)

split <- sample.split((1:nrow(train_df)), SplitRatio = 0.7)
train <- train_df[split,]
valid <- train_df[!split, ]

varnames <- setdiff(names(train), c("listing_id", "cat"))
features <- setdiff(varnames, 'interest_level')
#train1 <- as.data.frame(train)
train.hex <- as.h2o(train)

h2o_gbm <- h2o.gbm(x = features, y = "interest_level", train.hex, seed = 10000)
#h2o_gbm <- h2o.gbm(1:5, 6, train.hex, seed = 10000)

true_interest <- valid$interest_level
valid1 <- valid %>%
  select(-interest_level)
valid.hex <- as.h2o(valid1)
predH2o <- as.data.frame(h2o.predict(h2o_gbm, valid.hex))
pred <- predH2o$predict
table(true_interest, pred)

valid_result1 <- cbind(valid, predH2o)
valid_result <- valid_result1 %>%
  select(listing_id, interest_level, predict, high, medium, low) %>%
  mutate(totalProb = high + medium + low)

######################
test_df <- final_df %>%
  filter(cat == 'test') %>%
  select(-c(cat, interest_level))

trainAll.hex <- as.h2o(train_df)

h2o_gbm <- h2o.gbm(x = features, y = "interest_level", trainAll.hex, seed = 10000)


test.hex <- as.h2o(test_df)
predH2o <- as.data.frame(h2o.predict(h2o_gbm, test.hex))
test_result1 <- cbind(test_df, predH2o)
test_result <- test_result1 %>%
  select(listing_id, high, medium, low)
write.csv(test_result, file = 'submission1.csv', quote = FALSE, row.names = FALSE)


###############
h2o_rf <- h2o.randomForest(x = features, y = "interest_level", trainAll.hex, seed = 10000)
predH2o <- as.data.frame(h2o.predict(h2o_rf, test.hex))
test_result1 <- cbind(test_df, predH2o)
test_result <- test_result1 %>%
  select(listing_id, high, medium, low)
write.csv(test_result, file = 'submission2.csv', quote = FALSE, row.names = FALSE)
