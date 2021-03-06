setwd("C:\\Users\\alin\\Documents\\SelfStudy\\MyLearning\\FDA2018")
library('ReporteRs')
library(flextable)
system("java -version")



# download.file(url="http://www.sthda.com/sthda/RDoc/example-files/r-reporters-powerpoint-template.pptx",
#               destfile="r-reporters-powerpoint-template.pptx", quiet=TRUE)

options('ReporteRs-fontsize'= 18, 'ReporteRs-default-font'='Arial')
#doc <- pptx(template="r-reporters-powerpoint-template.pptx" )
doc <- pptx(template = 'facet.pptx')
#doc <- pptx(template="A_template.pptx")
# Slide 1 : Title slide
#+++++++++++++++++++++++
doc <- addSlide(doc, "Title Slide")
doc <- addTitle(doc,"Applications of Statistics in Education Business")
doc <- addSubtitle(doc, "Anhua Lin")
doc <- addDate(doc)
#doc <- addFooter(doc, "Anhua Lin")
#doc <- addPageNumber(doc, "1/4")

#writeDoc(doc, "FDA_Presentation.pptx" )

# Slide 2: Background
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Background Information")
doc  <- addParagraph(doc, 
                     value = c('Business requests', "Similarity with clinical trials"),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Slide 3: Two types of problem
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Two Types of Problems")
doc  <- addParagraph(doc, 
                     value = c('Evaluating new approaches', 
                               "Providing analytical guidence"),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Slide 3.5: Workflow
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Typical Work Flow")
doc  <- addParagraph(doc, 
                     value = c('Discuss with business contact', 
                               'Collect and clean data (SAS, SQL)',
                               'Analyze data (R)',
                               'If needed, build model(s) (R, Python)'),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)
# Slide 4: Case Study 1
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Case Study I")
doc  <- addParagraph(doc, 
                     value = c('Context', 
                               'Request'
                               ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Slide 5: Data
# +++++++++++++++++++++
doc <- addSlide(doc, "Two Content")
doc <- addTitle(doc, "Data")
dtab <- vanilla.table(dat[1:10,c('id','tc','ret')])
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)
doc  <- addParagraph(doc, 
                     value = c('Treatment: new students in June 2017', 
                               'Control: new students in June 2016',
                               'Goal: improve second term retention',
                               'Not real data'
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Slide : Data
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Data")
dtab <- vanilla.table(dat[1:10,c('id', xvars[1:4])])
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)

# Slide : Data
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Data")
dtab <- vanilla.table(dat[1:10,c('id', xvars[5:8])])
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)

# Slide : Data
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Data")
dtab <- vanilla.table(dat[1:10,c('id', xvars[9:12])])
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)

# Slide : Other confounders
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Risk")
doc  <- addParagraph(doc, 
                     value = c('Ignore other potential confounders' 
                               
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Silde  : Check balance

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance Before Matching")
doc <- addPlot(doc, function() print(display_prop(df = dat, target = 'prog', 
                                  category = c('Management', 'EDD'))))

# Slide: Check balance
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance Before Matching")
raw_text <- capture.output(check_balance(d = dat, fld_name = 'prog'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))

# Silde  : Check balance

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Blance Before Matching")
doc <- addPlot(doc, function() print(display_prop(df = dat, target = 'pay_plan', 
                                                  category = c('1_installment', '3_installment'))))

# Slide: Check balance
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance Before Matching")
raw_text <- capture.output(check_balance(d = dat, fld_name = 'pay_plan'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))

# Silde  : Check balance

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance Before Matching")
doc <- addPlot(doc, function() print(display_prop(df = dat, target = 'honesty', 
                                                  category = c('No', 'Yes'))))

# Slide: Check balance
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance Before Matching")
raw_text <- capture.output(check_balance(d = dat, fld_name = 'honesty'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))


# Slide: standarized mean difference
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Standardized Mean Difference")
doc  <- addParagraph(doc, 
                     value = c('SMD is the difference in means between groups, divided by (pooled) standard deviation',
                               'Value < 0.1 indicates adequate balance',
                               'Value 0.1 - 0.2 are not too alarming',
                               'Value > 0.2 indicates serious imbalance'
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)


tab1_pre <- read.csv(file = 'table1_prematch.csv')

# Slide : prematch table1
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "SMD Before Matching")
dtab <- vanilla.table(tab1_pre)
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)

# Slide : PSM
# +++++++++++++++++++++
doc <- addSlide(doc, "Two Content")
doc <- addTitle(doc, "Propensity Score")
doc  <- addParagraph(doc, 
                     value = c('Estimate propensity scores by model', 
                               'Usually use logistic regression',
                               'Also may use other models'
                               ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

r_code0 <- "psmodel <- glm(tc ~ . - id - ret, 
      family = binomial(), data = dat)

dat$ps <- psmodel$fitted.values"
doc <- addRScript(doc, text=r_code0)

### Slide : compare common support
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Common Support")
# doc <- addPlot(doc, function() print(display_prop(df = dat, target = 'honesty', 
#                                                   category = c('No', 'Yes'))))

doc <- addPlot(doc, function() commonsupport())

# doc <- addSlide(doc, "Two Content")
# doc <- addTitle(doc, "Propensity Score Matching")
# doc  <- addParagraph(doc,
#                      value = c('Match on the logit of propensity scores',
#                                'May try greedy match and optimal match and different caliper',
#                                'The goal is to balance confounders across treatment and control groups',
#                                'There are both arguments  in favor of and arguments against PSM'),
#                      par.properties = parProperties(list.style = 'ordered', level = 1)
# )
# 
# r_code1 <- "logit <- function(p) {log(p)-log(1-p)}
# 
# psmatch <- Match(Tr=dat$tc, M=1, X=logit(dat$ps),replace=FALSE,caliper= 1.3)
# 
# matched<-dat[unlist(psmatch[c('index.treated','index.control')]), ]"
# 
# doc <- addRScript(doc, text=r_code1)



# Silde  : Check balance

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance After Matching")
doc <- addPlot(doc, function() print(display_prop(df = matched, target = 'prog', 
                                                  category = c('Management', 'EDD'))))

# Slide: Check balance
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance After Matching")
raw_text <- capture.output(check_balance(d = matched, fld_name = 'prog'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))

# Silde  : Check balance

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance After Matching")
doc <- addPlot(doc, function() print(display_prop(df = matched, target = 'pay_plan', 
                                                  category = c('1_installment', '3_installment'))))

# Slide: Check balance
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance After Matching")
raw_text <- capture.output(check_balance(d = matched, fld_name = 'pay_plan'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))

# Silde  : Check balance

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance After Matching")
doc <- addPlot(doc, function() print(display_prop(df = matched, target = 'honesty', 
                                                  category = c('No', 'Yes'))))

# Slide: Check balance
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Check Balance After Matching")
raw_text <- capture.output(check_balance(d = matched, fld_name = 'honesty'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))


# Slide: standarized mean difference

tab1_after <- read.csv(file = 'table1_aftermatch.csv')

# Slide : aftermatch table1
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "SMD After Matching")
dtab <- vanilla.table(tab1_after)
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)


# Slide: Outcome
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Outcome")
raw_text <- capture.output(check_balance(d = matched, fld_name = 'ret', alternative = 'greater'))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))

writeDoc(doc, "FDA_Presentation.pptx" )
