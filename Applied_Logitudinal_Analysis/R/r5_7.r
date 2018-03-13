library(foreign)
dir <- 'C:\\Users\\alin\\Documents\\SelfStudy\\MyLearning\\Applied_Logitudinal_Analysis\\data\\'
fpath <- paste(dir, 'tlc-data.txt', sep = '')
ds <- read.table(file = fpath)
names(ds) <- c('id', 'trt', 'y0', 'y1', 'y4', 'y6')
ds$baseline <- ds$y0
df <- reshape(ds, idvar="id", varying=c("y0","y1","y4","y6"), 
                   v.names="y", timevar="time", time=1:4, direction="long")
df <- subset(df, time > 1)
df$week <- df$time
df$week[df$time == 2] <- 1
df$week[df$time == 3] <- 4
df$week[df$time == 4] <- 6

df$time <- df$time - 1
df$week.f <- factor(df$week, c(1,4,6))
attach(df)
library(nlme)
model <- gls(y ~ I(week.f==1) + I(week.f==4) + I(week.f==6) +  
               I(week.f==1 & trt=="A") + I(week.f==4 & trt=="A") + 
              I(week.f==6 & trt=="A"),  corr=corSymm(, form= ~ time | id), 
              weights = varIdent(form = ~ 1 | week.f))