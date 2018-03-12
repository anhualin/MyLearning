library(foreign)
dir <- 'C:\\Users\\alin\\Documents\\SelfStudy\\MyLearning\\Applied_Logitudinal_Analysis\\data\\'
fpath <- paste(dir, 'tlc-data.txt', sep = '')
ds <- read.table(file = fpath)
names(ds) <- c('id', 'trt', 'y0', 'y1', 'y4', 'y6')
df <- reshape(ds, idvar="id", varying=c("y0","y1","y4","y6"), 
          v.names="y", timevar="time", time=1:4, direction="long")

df$week <- df$time
df$week[df$time == 1] <- 0
df$week[df$time == 2] <- 1
df$week[df$time == 3] <- 4
df$week[df$time == 4] <- 6

interaction.plot(df$week, df$trt, df$y, type="b", pch=c(19,21), ylim=c(10, 30), 
                 xlab="Time (in weeks)", ylab="Blood Lead Levels", 
                 main="Plot of Mean Response Profiles in the Placebo & Succimer Groups", 
                 col=c(2,4))
df$week.f <- factor(df$week, c(0,1,4,6))
library(nlme)


model <- gls(model = y ~ trt*week.f, data = df, corr=corSymm(, form= ~ time | id), 
             weights = varIdent(form = ~ 1 | week.f))
summary(model)

anova(model)
