library(topicmodels)

data("AssociatedPress")
AssociatedPress

ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))