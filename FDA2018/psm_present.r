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

a <- display_prop(df = dat, target = 'prog', category = c('Management', 'EDD'))
check_balance(d = dat, fld_name = 'prog')
