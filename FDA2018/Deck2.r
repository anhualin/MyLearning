setwd("C:\\Users\\alin\\Documents\\SelfStudy\\MyLearning\\FDA2018")
library('ReporteRs')
library(flextable)
system("java -version")

options('ReporteRs-fontsize'= 18, 'ReporteRs-default-font'='Arial')
#doc <- pptx(template="r-reporters-powerpoint-template.pptx" )
doc <- pptx(template = 'facet.pptx')
# Slide 4: Case Study 2
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Case Study II")
doc  <- addParagraph(doc, 
                     value = c('Context', 
                               'Request'
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Slide: Data
# +++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Data")
doc  <- addParagraph(doc, 
                     value = c('10679 records in total', 
                               'Target variable: risk',
                               '26 raw features',  
                               'Randomly split into 70% train and 30% test'
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

# Slide: Data
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Data")
dtab <- vanilla.table(train[1:10,c('id','risk', 'gender', 'ba_age', 'program','degree',  'cum_gpa')])
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)

# Slide: Data
# +++++++++++++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Data")
dtab <- vanilla.table(train[1:10,c('id', 'current_balance', 'in_dissertation', 'in_future')])
dtab <- setZebraStyle(dtab, odd = '#eeeeee', even = 'white')
#dtab <-  bg(dtab, bg = "#E4C994", part = "header")
#dtab <- align(dtab, align = "center", part = "all" )
doc <- addFlexTable(doc, dtab)

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "EDA (GPA): Double Density Plot")
doc <- addPlot(doc, function() print(ggplot(data=train) +
                                       geom_density(aes(x = cum_gpa,
                                                        color = risk,
                                                        linetype = risk))))
  
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "EDA (bb_gip): Missing Values")
raw_text <- capture.output(summary(train$bb_gip_etc_ratio_std))
my_text <- pot(trimws(paste(raw_text, collapse = '\n')))
doc <- addParagraph(doc, value = set_of_paragraphs(my_text),
                    par.properties=parProperties(text.align="justify"))
                                                  


# Slide: Data
# +++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Feature Selection")
doc  <- addParagraph(doc, 
                     value = c('Domain knowledge', 
                               'AUC of single variable model',
                               'AIC based on index on single variable model',  
                               'Other approaches',
                               'No guarantee, trial and error'
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "AUC (Program)")
doc <- addPlot(doc, function(){
  eval <- prediction(valid0[,'predprogram'], valid0[, outcome])
  plot(performance(eval,"tpr","fpr"))
  print(attributes(performance(eval,'auc'))$y.values[[1]])
  }
)
  
 
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "AUC (Credits)")
doc <- addPlot(doc, function(){
  eval <- prediction(valid0[,'predba_credits_passed_prior1yr'], valid0[, outcome])
  plot(performance(eval,"tpr","fpr"))
  print(attributes(performance(eval,'auc'))$y.values[[1]])
}
) 

# Slide: Models
# +++++++++++
doc <- addSlide(doc, "Two Content")
doc <- addTitle(doc, "Statistical Learning Models")
doc  <- addParagraph(doc, 
                     value = c('Logistic Regression', 
                               'Random Forest',
                               'Extra Trees',  
                               'Gradient Boosting',
                               'Neural Network',
                               'Ensembling and Stacking',
                               '...'
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)
doc <- addImage(doc, "C:\\Users\\alin\\Documents\\Data\\FDA2018\\ESL.jpeg")

# Slide: Data
# +++++++++++
doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Model Selection and Parameter Tuning")
doc  <- addParagraph(doc, 
                     value = c('Cross Validation', 
                               'Grid Search'
                               
                     ),
                     par.properties = parProperties(list.style = 'ordered', level = 1)
)

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Model Performance")

doc <- addSlide(doc, "Title and Content")
doc <- addTitle(doc, "Model Performance")

writeDoc(doc, "FDA_Presentation2.pptx" )