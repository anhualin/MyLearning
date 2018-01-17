require(tableone)
require(Matching)
require(Hmisc)       # (Frank Harrell's library of useful functions)
require(dplyr)      # for data manipulation
require(tidyr)      # for reshaping data
require(ggplot2)
require(scales)
require(ggthemes) 

dat <- readRDS("C:\\Users\\alin\\Documents\\SelfStudy\\CausalEffectsVideos\\psm.rds")

display_prop <- function(df, tc_fld = 'tc', target, category){
  # show bar charts of proportion of a binary variate by  treatment/control
  trt_cnt <- sum(df[tc_fld])
  ctl_cnt <- sum(1 - df[tc_fld])
  df['treat'] <- as.factor(ifelse(df[tc_fld] == 0, 'control', 'treatment'))
  df['grp'] <- as.factor(ifelse(df[target] == 0, category[1], category[2]))
  df_prop <- df %>%
    group_by(treat, grp) %>%
    summarise(freq = n()) %>%
    mutate(prop = ifelse(treat == 'control', freq / ctl_cnt, freq / trt_cnt))
  
  
  ggplot(data = df_prop, aes(x = treat, y = prop, fill = grp)) +
    geom_bar(stat = 'identity', position = 'dodge', alpha = 2/3) +
    scale_y_continuous(labels = percent) +
    scale_fill_few('medium', drop = FALSE) +
    labs(x = NULL, y = 'Proportion', fill = target,
         title = paste('Proportion of ', target))
}

check_balance <- function(d, test_type = 'p', trt_name='tc', fld_name, alternative = 'two.sided', paired = FALSE){
  # use t-test (for numerical variate) and z-test (for categorical variate) to check whether 
  # the means / proportions of the given variate are the same for treatment and control
  trt <- d[d[trt_name] == 1, fld_name]
  ctl <- d[d[trt_name] == 0, fld_name]
  
  if(test_type == 'p'){
    prop.test(c(sum(trt), sum(ctl)), c(length(trt), length(ctl)), alternative = alternative)
  }else{
    t.test(ctl, trt, alternative = alternative)
  }
}

# a <- display_prop(df = dat, target = 'prog', category = c('Management', 'EDD'))
# b <- capture.output(check_balance(d = dat, fld_name = 'prog'))
# g <- trimws(paste(b, collapse = '\n'))
# check_balance(d = dat, fld_name = 'prog')

table1 <- CreateTableOne(vars = xvars, strata = 'tc', data = dat, test = FALSE)
print_table1 <- print(table1, smd = TRUE, exact = "stage", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
write.csv(print_table1, file = 'table1_prematch.csv')

psmodel <- glm(tc ~ gender + age + prog + balance + honesty + pay_plan 
               + assignment + forum + africa + asia
               + europe + america + other_region, 
               family = binomial(), data = dat)


dat$ps <- psmodel$fitted.values

commonsupport <- function(){
  plot(dat$ps, jitter(dat$tc+0.5), pch=16, cex=0.5, col=2*dat$tc+2,
       axes=FALSE, type="p", xlim=c(0,1), ylim=c(0,2), xlab="", ylab="")
  axis(1)
  mtext(c("control", "treatment"), side=2, at=c(0.5, 1.5), line=3, las=1, adj=0)
  mtext("Propensity score", side=1, at=0.5, line=2.5)
  lines(quantile(dat$ps[dat$tc==0])[c(2,4)], c(0.85, 0.85), col=2)
  lines(rep(quantile(dat$ps[dat$tc==0])[3], 2), c(0.8, 0.9), col=2)
  lines(quantile(dat$ps[dat$tc==1])[c(2,4)], c(1.15, 1.15), col=4)
  lines(rep(quantile(dat$ps[dat$tc==1])[3], 2), c(1.1, 1.2), col=4)
}

logit <- function(p) {log(p)-log(1-p)}
psmatch<-Match(Tr=dat$tc, M=1,X=logit(dat$ps),replace=FALSE,caliper= 1.3)
matched<-dat[unlist(psmatch[c("index.treated","index.control")]), ]

table1 <- CreateTableOne(vars = xvars, strata = 'tc', data = matched, test = FALSE)
print_table1 <- print(table1, smd = TRUE, exact = "stage", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
write.csv(print_table1, file = 'table1_aftermatch.csv')
