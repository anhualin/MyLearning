library(jsonlite)
metadata <- fromJSON("https://data.nasa.gov/data.json")
names(metadata$dataset)

library(dplyr)

nasa_title <- data_frame(id = metadata$dataset$`_id`$`$oid`, 
                         title = metadata$dataset$title)
nasa_title

nasa_desc <- data_frame(id = metadata$dataset$`_id`$`$oid`, 
                        desc = metadata$dataset$description)

nasa_desc %>% 
  select(desc) %>% 
  sample_n(5)

library(tidyr)

nasa_keyword <- data_frame(id = metadata$dataset$`_id`$`$oid`, 
                           keyword = metadata$dataset$keyword) %>%
  unnest(keyword)

nasa_keyword


library(tidytext)

nasa_title <- nasa_title %>% 
  unnest_tokens(word, title) %>% 
  anti_join(stop_words)

nasa_desc <- nasa_desc %>% 
  unnest_tokens(word, desc) %>% 
  anti_join(stop_words)

nasa_title %>%
  count(word, sort = TRUE)

nasa_desc %>% 
  count(word, sort = TRUE)


my_stopwords <- data_frame(word = c(as.character(1:10), 
                                    "v1", "v03", "l2", "l3", "l4", "15", "v5.2.0", 
                                    "v003", "v004", "v005", "v006", "v7"))
nasa_title <- nasa_title %>% 
  anti_join(my_stopwords)
nasa_desc <- nasa_desc %>% 
  anti_join(my_stopwords)

nasa_keyword %>%
  count(keyword, sort = TRUE)

nasa_keyword <- nasa_keyword %>%
  mutate(keyword = toupper(keyword))

library(widyr)

title_word_pairs <- nasa_title %>% 
  pairwise_count(word, id, sort = TRUE, upper = FALSE)
