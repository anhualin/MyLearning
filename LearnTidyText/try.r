library(dplyr)
text <- c("Because I could not stop for Death -",
          "He kindly stopped for me -",
          "The Carriage held but just Ourselves -",
          "and Immortality")

text_df <- data_frame(line = 1:4, text = text)

library(tidytext)
text_df %>%
  unnest_tokens(word, text)
