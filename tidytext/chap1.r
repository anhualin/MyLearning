#unnest_tokens()
text <- c("Because I could not stop for Death -",
          "He kindly stopped for me -",
          "The Carriage held but just Ourselves -",
          "and Immortality")

text

library(dplyr)

#detour
#basics of dplyr
library(nycflights13)
dim(flights)
head(flights)

filter_out <- filter(flights, month == 1, day == 1)

#To select rows by position, use slice():
slice_out <- slice(flights, 1:2) 
  
arrange_out <- arrange(flights, year, month, day)
arrange_out1 <- arrange(flights, year, desc(month), day)

select_out <- select(flights, year, month, day)
select_out1 <-select(flights, -(year: day))


#There are a number of helper functions you can use within select(), like starts_with(), ends_with(), matches() and contains(). These let you quickly match larger blocks of variables that meet some criterion. See ?select for more details.

#You can rename variables with select() by using named arguments:

select_out2 <- select(flights, deptime = dep_time)
#rename is better
select_out3 <- rename(flights, deptime = dep_time)

#Use distinct()to find unique values in a table:
distinct(flights, tailnum)
distinct(flights, origin, dest)

#Add new columns with mutate()
mutate(flights, gain = arr_delay - dep_delay, speed = distance / air_time * 60)

# group function
by_tailnum <- group_by(flights, tailnum)
delay <- summarise(by_tailnum,
                   count = n(),
                   dist = mean(distance, na.rm = TRUE),
                   delay = mean(arr_delay, na.rm = TRUE))
delay <- filter(delay, count > 20, dist < 2000)


delay1 <- flights %>% 
    group_by(tailnum) %>%
    summarise(count = n(), 
              dist = mean(distance, na.rm = TRUE),
              delay = mean(arr_delay, na.rm = TRUE)) %>% 
    filter(count > 20, dist < 2000)
  
library(ggplot2)
ggplot(delay, aes(dist, delay)) +
  geom_point(aes(size = count), alpha = 1/2) +
  geom_smooth() +
  scale_size_area()

ggplot(delay1, aes(dist, delay)) +
  geom_point(aes(size = count), alpha = 1/2) +
  geom_smooth() +
  scale_size_area()


#summarize, group by
result <- flights %>%
  group_by(dest) %>%
  summarise(plane = n_distinct(tailnum), flights = n())
    

# back to tidytext
text_df <- data_frame(line = 1:4, text = text)
library(tidytext)

text_df %>% 
  unnest_tokens(word, text)

library(janeaustenr)
#library(dplyr)
library(stringr)

original_books <- austen_books() %>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                 ignore_case = TRUE)))) %>%
  ungroup()

original_books

text <- c("Because I could not stop for Death -",
          "The Carriage held but just Ourselves -",
          "Chapter 1",
          "Chapter 1")
libro <- c("book1", "book2", "book1", "book2")

test_df <- data_frame(line = 1:4, book = libro, text = text)

test_books <- test_df %>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                 ignore_case = TRUE)))) %>%
  ungroup()

test_books

test_books1 <- test_df %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                 ignore_case = TRUE)))) 

test_books1



tidy_books <- original_books %>%
  unnest_tokens(word, text)

data("stop_words")
tidy_books <- tidy_books %>%
  anti_join(stop_words)

tidy_books %>%
  count(word, sort = TRUE)

tidy_books %>%
  count(word, sort = TRUE) %>%
  filter(n > 600) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) + 
  geom_col() + 
  xlab(NULL) +
  coord_flip()


library(gutenbergr)
workingSite <- 'ftp://ftp.ibiblio.org/pub/docs/books/gutenberg'

hgwells <- gutenberg_download(c(35,36, 5230, 159),
                              mirror = workingSite)

tidy_hgwells <- hgwells %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

tidy_hgwells %>%
  count(word, sort = TRUE)

bronte <- gutenberg_download(c(1260, 768, 969, 9182, 767), mirror = workingSite)
tidy_bronte <- bronte %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)

tidy_bronte %>%
  count(word, sort = TRUE)

library(tidyr)

frequency <- bind_rows(mutate(tidy_bronte, author = "Brontë Sisters"),
                       mutate(tidy_hgwells, author = "H.G. Wells"), 
                       mutate(tidy_books, author = "Jane Austen")) %>%
  mutate(word = str_extract(word, "[a-z']+")) %>%
  count(author, word) %>%
  group_by(author) %>%
  mutate(proportion = n / sum(n)) %>% 
  select(-n) %>% 
  spread(author, proportion) %>% 
  gather(author, proportion, `Brontë Sisters`:`H.G. Wells`)

library(scales)
ggplot(frequency, aes(x = proportion, y = `Jane Austen`, color = abs(`Jane Austen` - proportion))) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  facet_wrap(~author, ncol = 2) +
  theme(legend.position="none") +
  labs(y = "Jane Austen", x = NULL)

cor.test(data = frequency[frequency$author == "Brontë Sisters",],
         ~ proportion + `Jane Austen`)


