#install.packages(c('tm', 'SnowballC', 'wordcloud', 'topicmodels'))
library(tm)
library(SnowballC)
library(wordcloud)




if (Sys.info()[1] == 'Windows'){
  setwd('C:/Users/alin/Documents/SelfStudy/MyLearning/tidytext/')
}else{
  setwd('/home/alin/MyLearning/tidytext/')  
}

#https://github.com/dipanjanS/text-analytics-with-python/blob/master/Chapter-7/movie_reviews.csv

reviews = read.csv("movie_reviews.csv", stringsAsFactors = F)
reviews$content <- reviews$review
reviews$polarity <- as.factor(reviews$sentiment)
reviews$review <- NULL
reviews$sentiment <-NULL

review_corpus = Corpus(VectorSource(reviews$content))
review_corpus = tm_map(review_corpus, content_transformer(tolower))
review_corpus = tm_map(review_corpus, removeNumbers)
review_corpus = tm_map(review_corpus, removePunctuation)
review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
review_corpus =  tm_map(review_corpus, stripWhitespace)

inspect(review_corpus[1])

review_dtm <- DocumentTermMatrix(review_corpus)
review_dtm

inspect(review_dtm[500:505, 500:505])

review_dtm = removeSparseTerms(review_dtm, 0.95)
review_dtm

inspect(review_dtm[1,1:20])

findFreqTerms(review_dtm, 5000)

freq = data.frame(sort(colSums(as.matrix(review_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))

review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf

inspect(review_dtm_tfidf[1,1:20])

freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))
