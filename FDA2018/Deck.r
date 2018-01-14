library('ReporteRs') 
system("java -version")
doc <- pptx()
slide.layouts(doc)

layouts <-slide.layouts(doc) # All available layout
#  plot each slide style
for(i in layouts ){
  par(mar=c(0.5,0.5,2,0.5), cex=0.7)
  slide.layouts(doc, i )
  title(main = paste0("'", i, "'" ))
  if(interactive()) readline(prompt = "Show next slide layout")
}

