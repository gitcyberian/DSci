if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ModelMetrics)) install.packages("ModelMetrics", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(rattle)) install.packages("rattle", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")

# Bank Marketing dataset:
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip", dl)

data <- fread(text = gsub(";", "\t", readLines(unzip(dl, "bank-additional/bank-additional-full.csv"))), stringsAsFactors = TRUE)
#setwd('C:\\vivek\\Rcourse\\datascience\\Capstone2\\bank-deposit\\')
#data <- fread(text = gsub(";", "\t", readLines(con=file('bank-additional-full.csv'))), stringsAsFactors = TRUE)
#We read all character fields as factors since the dataset description mentions they are all categorical.

#Let's first change the target y to 0/1 instead of no/yes
data$y <- ifelse(data$y=='yes',1,0)

#Let's split the data set into an analysis data set and a validation set.
#All data exploration, training and testing and tuning will happen using the analysis set. The validation set will be 
#used for the final evaluation of the model chosen.
set.seed(1, sample.kind="Rounding")
idx <- createDataPartition(y=data$y, times=1, p = 0.1, list = FALSE)
validation <- data[idx,]
data <- data[-idx,]

nDataRows<-NROW(data)
nvalidRows<-NROW(validation)
#We have nDataRows rows in the analysis data set and nvalidRows in the validation set.

#We want to build two prediction models, one that includes duration as a feature and one that does not

#We have nDataRows rows in the analysis data set and nvalidRows in the validation set.

#Let's have a quick look at the structure of the dataset and a few samples from the data
str(data)
head(data)

#We have the description of the columns from the dataset providers:
#1 - age (numeric)
#2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
#3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
#4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
#5 - default: has credit in default? (categorical: 'no','yes','unknown')
#6 - housing: has housing loan? (categorical: 'no','yes','unknown')
#7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
#8 - contact: contact communication type (categorical: 'cellular','telephone') 
#9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
#10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
#11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
#12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#14 - previous: number of contacts performed before this campaign and for this client (numeric)
#15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
#16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
#18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
#19 - euribor3m: euribor 3 month rate - daily indicator (numeric). This is the 3 months Euro Interbank Offered Rate
#20 - nr.employed: number of employees - quarterly indicator (numeric)

#Output variable (desired target):
#21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

#Let's look at the distribution of positive and negative cases.
data %>% group_by(y) %>% summarize(accept_count=n())
mean(data$y)

#We can see this is an imbalanced dataset with only about 11% of positive records.

#Let's first look at the distribution of values for each column.
data %>% ggplot(aes(age)) + geom_histogram(bins=10) + scale_x_continuous(breaks=seq(10, 100, 5))
#We see majority of the dataset between the age of 23 and 58 - this is expected as banks tend to approach customers 
#who are in the working age range.
#How does the target vary by age?
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=age, color=y)) + geom_density() + scale_x_continuous(breaks=seq(10, 100, 5))
#We don't see much of a variation here, except that people over 58 are more likely to accept and this is intuitive as 
#older/retired people tend to prefer relatively safer investment avenues such as term deposits.

#Let's look at job now.
data %>% ggplot(aes(job)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We can see bulk of the people are admin or blue-collared or technician.
#How does the target vary by job?
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=job, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(job) %>% summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=job, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We can see that the proportion of acceptances varies by job. Students and retired people are more likely to accept 
#than others.

#Let's look at marital status now.
data %>% ggplot(aes(marital)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We can see the majority of our dataset are married.
#How does the target vary by marital status?
data %>% mutate(y=factor(y)) %>% filter(marital!='unknown') %>% ggplot(aes(x=marital, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#Since there are very few in the unknown category, let's filter that out while comparing the accept %.
data %>% filter(marital!='unknown') %>% group_by(marital) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=marital, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We don't see much variation here. Single people have a slightly higher percentage of acceptances.

#Let's look at education now.
data %>% ggplot(aes(education)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We can see very few illiterate cases. Let's filter them out and check the acceptance by education.
#How does the target vary by education status?
data %>% mutate(y=factor(y)) %>% filter(education!='illiterate') %>% 
  ggplot(aes(x=education, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% filter(education!='illiterate') %>% group_by(education) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=education, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We see a slight variation here with customers with university degree or unknown having a few percentage points more.

#Let's look at default status now.
data %>% ggplot(aes(default)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
#There are hardly any default=yes cases. How many actually are there?
data %>% group_by(default) %>% summarize(count=n())
#Since there are only 3 yes cases, let's filter them out and compare only the "no" and the "unknown" cases.
data %>% mutate(y=factor(y)) %>% filter(default!='yes') %>% 
  ggplot(aes(x=default, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% filter(default!='yes') %>% group_by(default) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=default, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We see some variation here. People whose default status is unknown have a lower acceptance rate.

#Let's look at housing loan status now.
data %>% ggplot(aes(housing)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=housing, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(housing) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=housing, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(housing) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1))
#We see almost no variation here. So perhaps, the acceptance does not depend much on housing loan status.

#Let's look at personal loan status now.
data %>% ggplot(aes(loan)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(loan) %>% summarize(count=n())
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=loan, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(loan) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=loan, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(loan) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1))
#We see almost no variation here too. So perhaps, the acceptance does not depend much on personal loan status.

#Let's look at contact now.
data %>% ggplot(aes(contact)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=contact, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(contact) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=contact, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(contact) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1))
#We see a significant variation here. People contacted on their mobiles are more likely to accept than the ones 
#contacted by telephone.

#Let's look at last month contact now.
#Change the month column to a factor so that the x axis gets ordered by month correctly in the plots
months <- c('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
data$month <- factor(data$month, levels=months)

#a <- c('apr', 'aug', 'jan')
#class(a)
#a <- factor(a, levels=months)
#a[order(a)]
#months <- str_to_lower(month.abb)
#grep('apr', months)
#sapply(a, function (x){grep(x, months)})

data %>% ggplot(aes(month)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We see most of the contact in the past has happened in May and the summer months. Very few or none in winter (Dec-Mar)
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=month, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(month) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=month, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(month) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1))
#We see an interesting trend below. Customers last contacted in March/Sep/Oct/Dec seem to have a much higher acceptance rate. It may not be a coincidence that March, Sep and Dec are all quarter ending months.

#How about trend of acceptance by both mode and month of last contact?
data %>% group_by(contact, month) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=month, y=accept_pct)) + geom_bar(stat='identity') + 
  facet_wrap(~contact) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#That gives a pretty interesting result - for example, though the overall month-wise plot earlier showed a low % of 
#around 10% in June, customers contacted in June by cellular have a much higher acceptance % at 44%.

#Let's check the trend of acceptance by day of week.
days <- c('sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat')
data$day_of_week <- factor(data$day_of_week, levels=days)

data %>% ggplot(aes(day_of_week)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
#The data is evenly spread across all weekdays.
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=day_of_week, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(day_of_week) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=day_of_week, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(day_of_week) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1))
#Not much of variation by day_of_week. The data seems to be evenly spread.
#Maybe we can check if day_of_week + contact gives any different trend.
data %>% group_by(contact, day_of_week) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=day_of_week, y=accept_pct)) + geom_bar(stat='identity') + 
  facet_wrap(~contact) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#No variation here either.

#We can check if day_of_week + month gives any different trend.
data %>% group_by(month, day_of_week) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=day_of_week, y=accept_pct)) + geom_bar(stat='identity') + 
  facet_wrap(~month) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We can see some variation now for March, April, Sep and Dec where the day of week impacts the acceptance %.
#But let's make sure these are not just outliers due to small counts.
data %>% filter(month %in% c('mar', 'apr', 'sep', 'dec')) %>% group_by(month, day_of_week) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=day_of_week, y=count)) + geom_bar(stat='identity') + 
  facet_wrap(~month) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#Based on the above, there is a moderate number of records for Mar, Apr (especially) and Sep, though not for Dec.
#It looks like day_of_week does play a role when taken in conjunction with month though not by itself.

#Let's look the trend by campaign now.
min(data$campaign)
max(data$campaign)
#We see that number of contacts performed in campaign ranges from 1 to 56.
data %>% ggplot(aes(campaign)) + geom_histogram(bins=10) + scale_x_continuous(breaks=seq(0, 60, 5))
#We see majority of the dataset have campaign falling in the range 1 to 10
#How does the target vary by campaign? Let's generate 2 plots, 1 for campaign<=10 and 1 for >10 to clearly see the trend.
data %>% mutate(y=factor(y)) %>% filter(campaign<=10) %>% 
  ggplot(aes(x=campaign, color=y)) + geom_density() + scale_x_continuous(breaks=seq(0, 10, 1))
data %>% mutate(y=factor(y)) %>% filter(campaign>10) %>% 
  ggplot(aes(x=campaign, color=y)) + geom_density() + scale_x_continuous(breaks=seq(10, 60, 5))
#The second plot shows that people contacted more than 15-20 times are very likely to reject.

#Let's try pdays now.
data %>% ggplot(aes(pdays)) + geom_histogram(bins=10)
#Looks like most of the data has pdays=999 which means they were never contacted before. Let's see the actual counts.
data %>% summarize(mean(pdays!=999))
#Only 3.6% of the data has pdays!=999. We can ignore this field as it will not give any meaningful analysis.
#Still, let's look at this 3.6% of the data and see how the acceptance count varies.
data %>% mutate(y=factor(y)) %>% filter(pdays!=999) %>% ggplot(aes(x=pdays, color=y)) + geom_density()

#The same is likely for "previous" too since it is non-zero only when pdays!=999
data %>% ggplot(aes(previous)) + geom_histogram(bins=10)
data %>% summarize(mean(previous!=0))
#That's weird since the dataset description defines previous as number of contacts performed before this campaign and 
#for this client. So if pdays=999 which means the client was never contacted before, then previous should be 0.
#However, we see 13% of records that have previous !=0 while pdays=999 only for 3.6%.
data %>% filter(pdays==999 & previous!=0) %>% group_by(previous) %>% summarize(count=n())
#Looks like previous=1 for most of the pdays=999 cases, so perhaps, this is an error in the description of the field.
data %>% mutate(y=factor(y)) %>% filter(previous!=0) %>% ggplot(aes(x=pdays, color=y)) + geom_density()
#This trend should probably be taken with a pinch of salt since we saw above that there is very little data for previous>1
#So, we should probably ignore this field.

#Let's look at poutcome now.
data %>% ggplot(aes(poutcome)) + geom_histogram(stat='count') + theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(poutcome) %>% summarize(count=n())
data %>% mutate(y=factor(y)) %>% ggplot(aes(x=poutcome, fill=y)) + geom_bar(position=position_dodge()) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
data %>% group_by(poutcome) %>% 
  summarize(count=n(), accept_pct=100*mean(y==1)) %>% 
  ggplot(aes(x=poutcome, y=accept_pct)) + geom_bar(stat='identity') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#We can clearly see that people who had previously accepted are likely to accept this time too.

#Let's now look at the social and economic context fields.

#Let's look at the employment variation rate
unique(data$emp.var.rate)
g1 <- data %>% ggplot(aes(emp.var.rate)) + geom_histogram(bins=10) + scale_x_continuous(breaks=seq(-4, 1.5, 0.5))
#How does the target vary by emp.var.rate?
g2 <- data %>% mutate(y=factor(y)) %>% 
  ggplot(aes(x=emp.var.rate, color=y)) + geom_density(show.legend = FALSE) + scale_x_continuous(breaks=seq(-4, 1.5, 0.5))
#We can see a distinguishing trend here with negative values of employment variation rate having a higher trend of acceptance compared to positive values.
grid.arrange(g1, g2, nrow=2)

#Let's look at the consumer price index
g1 <- data %>% ggplot(aes(cons.price.idx)) + geom_histogram(bins=10)
#We see majority of the data has consumer price index between 93 and 94.5 with very few records outside of this range.
#How does the target vary by cons.price.idx?
g2 <- data %>% mutate(y=factor(y)) %>% ggplot(aes(x=cons.price.idx, color=y)) + geom_density(show.legend=FALSE)
grid.arrange(g1, g2, nrow=2)
#There are specific small ranges where we see a slightly higher proportion of acceptances.

#Let's look at the consumer confidence index.
g1 <- data %>% ggplot(aes(cons.conf.idx)) + geom_histogram(bins=10)
#How does the target vary by consumer confidence index?
g2 <- data %>% mutate(y=factor(y)) %>% ggplot(aes(x=cons.conf.idx, color=y)) + geom_density(show.legend = FALSE)
grid.arrange(g1, g2, nrow=2)
#We can see a distinguishing trend here with consumer conf index in the range of -40 to -37.5 and >-35.

#Let's look at the euribor3m rate.
g1 <- data %>% ggplot(aes(euribor3m)) + geom_histogram()
#How does the target vary by euribor3m?
g2 <- data %>% mutate(y=factor(y)) %>% ggplot(aes(x=euribor3m, color=y)) + geom_density(show.legend = FALSE)
grid.arrange(g1, g2, nrow=2)
#We can see a distinguishing trend here with low values of euribor3m having more of the acceptance cases.

#Let's look at the number of employees now.
g1 <- data %>% ggplot(aes(nr.employed)) + geom_histogram(bins=10)
#How does the target vary by nr.employed?
g2 <- data %>% mutate(y=factor(y)) %>% ggplot(aes(x=nr.employed, color=y)) + geom_density(show.legend = FALSE)
grid.arrange(g1, g2, nrow=2)
#We can see a distinguishing trend here as well with <5100 employees having a higher acceptance ratio.

#Is there a correlation between these social and economic context fields and the acceptance?
temp <- data %>% select(emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed, y)
cor(temp)[1:5,6]
#Based on the correlation coefficients, we can see that emp.var.rate, euribor3m and nr.employed have a 
#moderately high correlation and that is what we saw in our plots above as well.

#So, let's summarise what we have observed so far.
#Age - slight variation. People above 58.
#Job - shows variation esp for retired and student
#Marital status - not much variation
#Education - slight variation here with customers with university degree or unknown having a few percentage points more.
#Previous Defaulters - some variation here. People whose default status is unknown have a lower acceptance rate.
#Housing loan status - no variation
#Personal loan status - no variation
#Contact mode - Significant variation here. People contacted on their mobiles are more likely to accept than the ones contacted by telephone.
#Last contact month - Customers last contacted in March/Sep/Oct/Dec seem to have a much higher acceptance rate. customers contacted in June by cellular have a much higher acceptance % at 44% than if they were contacted by landline.
#day_of_week - not much variation here.
#day of week plus month - has variation for April and to a lesser extent March and Sep where the day of week impacts the acceptance %.
#Number of times contacted - people contacted more than 15-20 times are very likely to reject.
#pdays and previous - can be ignored as they don't show much selectivity
#poutcome - People who had previously accepted are likely to accept this time too.
#emp.var.rate, euribor3m and nr.employed - all have a bearing on the acceptance
#The consumer conf index shows a minor distinguishing trend in specific ranges of -40 to -37.5 and >-35.

#We can now try a few prediction models.
#Let's first split the data set into training and test sets.
duration <- data[,'duration']
data <- data[,-'duration']

idx <- createDataPartition(y=data$y, times=1, p = 0.2, list = FALSE)
test <- data[idx,]
train <- data[-idx,]
duration_test <- duration[idx]
duration_train <- duration[-idx]

#Set a benchmark with a naive model that just predicts no/0 for the target (since it is a heavily imbalanced dataset).
naive_benchmark <- mean(test$y==0)
naive_benchmark
#How about sensitivity and specificity and the F1 score for the naive model.
pred <- rep(0, length(test$y))
#(caret::confusionMatrix(data=pred, reference=factor(test$y, levels=c(0,1))))$byClass[c('Sensitivity', 'Specificity', 'Balanced Accuracy')]
model_results <- data.frame(model='Naive benchmark', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
           specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results %>% knitr::kable()
#As can be seen, though we have specificity=1, the sensitivity=0 since we are always predicting 0 or "no".
#F_meas(data = pred, reference = test$y)

#Any model should perform better than this one for it to be relevant.
#We saw earlier age>58 may accept. Let's try that without using any machine learning.
pred <- ifelse(test$age>58, 1, 0)
res <- data.frame(model='Age rule', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
           specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This overall accuracy is not as good as the naive benchmark but we see an improvement in sensitivity now.

#Let's try by job
pred <- ifelse(test$job %in% c('student', 'retired'), 1, 0)
res <- data.frame(model='Job rule', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#Though the overall accuracy is still lower than the naive benchmark, the sensitivity shows further improvement.

#Let's try by age and job.
pred <- ifelse(test$age > 58 & test$job %in% c('student', 'retired'), 1, 0)
res <- data.frame(model='Age+Job rule', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#Overall accuracy improves, but the sensitivity reduces.

#Let's try one final rule model based on contact mode and month of contact:
pred <- ifelse(test$contact == 'cellular' & test$month %in% c('mar', 'jun', 'sep', 'dec'), 1, 0)
res <- data.frame(model='Contact+Month rule', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#We see a marked improvement in the sensitivity now at 0.19 and balanced accuracy at 58%.

#Let's try machine learning models now. The metric that we will train for will be the sensitivity.
specSummary <- function(data, lev=NULL, model=NULL) {
  tp <- sum(data$obs == data$pred & data$obs == 1)
  fp <- sum(data$obs != data$pred & data$pred == 1)
  fn <- sum(data$obs != data$pred & data$pred == 0)
  tn <- sum(data$obs == data$pred & data$obs == 0)
  #sens <- sensitivity(data$obs, data$pred)
  #spec <- ModelMetrics::specificity(data$obs, data$pred)
  #out <- (sens + spec)/2
  #out <- caret::confusionMatrix(data=data$pred, reference=data$obs)$byClass['Specificity']
  #out <- (sensitivity(data$obs, data$pred) + specificity(data$obs, data$pred))/2
  out <- (tp/(tp+fn) + tn/(tn+fp))/2
  names(out) <- 'bal_acc'
  out
}
metricControl <- trainControl(summaryFunction = specSummary)

#Let's try a simple logistic regression model with just using age.
age_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~age, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(age_lr, newdata=test)==1,1,0)
res <- data.frame(model='Age LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This is exactly the same as the naive benchmark and this models always predicts 'no'.

#Let's try based on job.
job_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~job, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(job_lr, newdata=test)==1,1,0)
res <- data.frame(model='Job LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This is exactly the same as the naive benchmark and this models always predicts 'no'.

#Let's try based on month.
month_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~month, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(month_lr, newdata=test)==1,1,0)
res <- data.frame(model='Month LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This is a little better now but still has low sensitivity and is not as good as the contact+month rule.

#Let's try based on month and contact mode.
month_contact_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~month+contact, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(month_contact_lr, newdata=test)==1,1,0)
res <- data.frame(model='Month+Contact LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This is better than the naive benchmark both in terms of accuracy and sensitivity, but the sensitivity is still low. It is still not as good as the contact+month rule.

#Let's try based on month and day_of_week.
month_day_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~month+day_of_week, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(month_day_lr, newdata=test)==1,1,0)
res <- data.frame(model='Month+day LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This is not as good as the previous LR models as sensitivity goes down to 0.06.

#Let's try based on month and day_of_week and contact mode.
month_day_contact_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~month+day_of_week+contact, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(month_day_contact_lr, newdata=test)==1,1,0)
res <- data.frame(model='Month+Contact+Day LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#The overall accuracy is better than all of the previous models including the rule-based models. The sensitivity has also improved over the previous LR models, but is still not as good as the simple month + contact rule model.

#Let's try a logistic regression model with all the relevant features that we found may be good influencers.
lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~age+job+education+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(lr, newdata=test)==1,1,0)
res <- data.frame(model='Selected Features LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#We see a good increase in overall accuracy, sensitivity and balanced accuracy. This is the best result so far. Sensitivity has gone up to 0.23 and balanced accuracy to 60%.

#Let's try a logistic regression model with all features.
lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~., data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(lr, newdata=test)==1,1,0)
res <- data.frame(model='All features LR model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This shows a marginal improvement over the previous model, but this may probably result in overfitting on the test set since we know some of the features are not good influencers.

#Let's try a gamLoess model now.
gammodel <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~age+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='gamLoess', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(gammodel, newdata=test)==1,1,0)
res <- data.frame(model='GAM Loess model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This model does not perform as well as the best lr model. The sensitivity is lower at 0.22 and balanced accuracy a little lower than 0.6.

#Let's try a multinomial model now.
multinomialmodel <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~age+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='multinom', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(multinomialmodel, newdata=test)==1,1,0)
res <- data.frame(model='Multinomial model', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This model is nearly as good but performs slightly lower than the best lr model.

#Let's try a naive Bayes model now.
nbmodel <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~age+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='naive_bayes', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(nbmodel, newdata=test)==1,1,0)
res <- data.frame(model='Naive Bayes', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This model is the best so far and performs much better than the other models. It gives a sensitivity of 0.42 and a balanced accuracy of 68%.
#Compare this with the naive benchmark of always predicting "no" which has 0 sensitivity and 50% balanced accuracy.

#Let's try a decision tree model now with the relevant features (with default parameters. We will tune the model later).
rpart_model <- train %>% mutate(y=factor(y, levels=c(0,1))) %>%
#rpart_model <- train %>% mutate(y=factor(ifelse(y=='yes',1,0), levels=c(0,1))) %>%
  train(y~age+job+education+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='rpart', metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(rpart_model, newdata=test)==1,1,0)
res <- data.frame(model='Decision Tree (def params)', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#The sensitivity is now at 0.25 and balanced accuracy at 0.62 which is better than the previous LR models but not as good as the naive_bayes model.
#But note that we have not yet tuned the hyperparameters for the decision tree.
fancyRpartPlot(rpart_model$finalModel)
#As we can see from the decision tree, the age, job and education fields are not used for the prediction.
#So, let's train a decision tree after excluding these fields. We should get the same level of accuracy.

rpart_model <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='rpart', metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(rpart_model, newdata=test)==1,1,0)
res <- data.frame(model='Decision Tree (def params)', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
res %>% knitr::kable()

fancyRpartPlot(rpart_model$finalModel, caption='Decision Tree')
#As we can see, the metrics are the same after removing those fields.

#Let's try to tune the decision tree model now.
#Since rpart does not allow tuning of multiple parameters in tuneGrid, we will have to manually iterate for minsplit.
minsplit = c(3, 6, 9, 12, 15, 18, 21)
senses <- {}
best_sens <- 0
best_i <- 0
best_model <- NULL
for(i in minsplit){
  rpart_model <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
    train(y~contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, 
                     data=., method='rpart', metric='bal_acc', trControl=metricControl,
                     tuneGrid=expand.grid(cp=seq(0, 0.05, len = 10)),
                     control = rpart.control(minsplit = i))
  pred <- ifelse(predict(rpart_model, newdata=test)==1,1,0)
  sens <- sensitivity(test$y, pred)
  senses <- rbind(senses, sens)
  if (best_sens < sens){
    best_sens <- sens
    best_i <- i
    best_model <- rpart_model
  }
}
data.frame(minsplit, senses) %>% ggplot(aes(minsplit, senses)) + geom_point() + geom_line()
best_i
best_sens

ggplot(best_model)

#fancyRpartPlot(best_model$finalModel)

pred <- ifelse(predict(best_model, newdata=test)==1,1,0)
tuned_rpart_acc <- mean(pred==test$y)
tuned_rpart_acc
res <- data.frame(model='Tuned Decision Tree', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()

#pruned_model <- prune(best_model$finalModel, cp=0.0007)
#fancyRpartPlot(pruned_model)

#Store the two best models for the first objective. This is for model aggregation later on.
obj1_top_1_model <- nbmodel#the naive bayes model
obj1_top_2_model <- best_model#the tuned decision tree model that was just trained above.

#Finally, we excluded the duration column since the objective of the case study was to see if we can only contact the 
#customers who are most likely to accept.
#However, given this is an extremely imbalanced daatset, for the completeness of this study, let's check if we can 
#make more accurate predictions if we also take duration into account. Such a model would still be useful in taking 
#key business actions as there may be a sizeable interval between the contact and the customer's decision.
train$duration <- duration_train
test$duration <- duration_test

duration_lr <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~., data=., method='glm', preProcess=c('center','scale'), metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(duration_lr, newdata=test)==1,1,0)
res <- data.frame(model='LR model with duration', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#We can see this LR model gives us a sensitivity of 0.4 and a balanced accuracy of 68%, with an overall accuracy of 90%.

#Let's build a naive bayes model now.
nbmodel <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~duration+age+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, 
        data=., method='naive_bayes', preProcess=c('center','scale'), 
        metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(nbmodel, newdata=test)==1,1,0)
res <- data.frame(model='Naive Bayes with duration', accuracy=mean(pred==test$y), 
                  sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#This naive bayes model that includes the duration gives a sensitivity of 0.46 and a balanced accuracy of 70%.


rpart_model <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
  train(y~duration+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=., method='rpart', metric='bal_acc', trControl=metricControl)
pred <- ifelse(predict(rpart_model, newdata=test)==1,1,0)
res <- data.frame(model='Decision Tree with duration', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()

fancyRpartPlot(rpart_model$finalModel, caption='Decision Tree')

#Let's try to tune the decision tree model that includes duration now.
#Since rpart does not allow tuning of multiple parameters in tuneGrid, we will have to manually iterate for minsplit.
minsplit = c(3, 6, 9, 12, 15, 18, 21)
senses <- {}
best_sens <- 0
best_minsplit_dur <- 0
best_model_dur <- NULL
for(i in minsplit){
  rpart_model <- train %>% mutate(y=factor(y, levels=c(0,1))) %>% 
    train(y~duration+contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, 
                       data=., method='rpart', metric='bal_acc', trControl=metricControl,
                       tuneGrid=expand.grid(cp=seq(0, 0.05, len = 10)),
                       control = rpart.control(minsplit = i))
  pred <- ifelse(predict(rpart_model, newdata=test)==1,1,0)
  sens <- sensitivity(test$y, pred)
  senses <- rbind(senses, sens)
  if (best_sens < sens){
    best_sens <- sens
    best_minsplit_dur <- i
    best_model_dur <- rpart_model
  }
}
data.frame(minsplit, senses) %>% ggplot(aes(minsplit, senses)) + geom_point() + geom_line()
best_minsplit_dur
best_sens

ggplot(best_model_dur)

pred <- ifelse(predict(best_model_dur, newdata=test)==1,1,0)
res <- data.frame(model='Tuned Decision Tree with duration', accuracy=mean(pred==test$y), sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()

#Store the two best models for the second objective. This is for model aggregation later on.
obj2_top_1_model <- best_model#the tuned decision tree model that was just trained above.
obj2_top_2_model <- nbmodel#the naive bayes model.

#Model aggregation for both objectives
#We already know that this dataset is a highly imbalanced one with very few positive samples in the dataset. Any machine learning model is very likely to underestimate the number of positive cases in the test set due to this class imbalance.
#One way we can improve the performance of prediction is to aggregate multiple prediction models and instead of averaging as is typically done in model aggregation, we can predict a postive outcome if either of the prediction models predicts a positive outcome.
#That way, we compensate for the underestimation of positive outcomes by each individual prediction model and this is likely to give a better performance.
#So, let's try this now for both the objectives one by one.
#Let's start with the two best models for the first objective. Predict 1 if either of the two models predicts 1.
pred1 <- ifelse(predict(obj1_top_1_model, newdata=test)==1,1,0)
pred2 <- ifelse(predict(obj1_top_2_model, newdata=test)==1,1,0)
pred <- ifelse(pred1==1 | pred2==1, 1, 0)

res <- data.frame(model='Final Aggregated Model for objective 1', 
                  accuracy=mean(pred==test$y), 
                  sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()
#We can see that these results are way better than each of the individual models. We get a sensitivity of `r round(res$sensitivity, 2)` and a balanced accuracy of `r res$balanced_accuracy * 100`%.

#Let's do the same for the two best models for the second objective. Predict 1 if either of them predicts 1.
pred1 <- ifelse(predict(obj2_top_1_model, newdata=test)==1,1,0)
pred2 <- ifelse(predict(obj2_top_2_model, newdata=test)==1,1,0)
pred <- ifelse(pred1==1 | pred2==1, 1, 0)

res <- data.frame(model='Final Aggregated Model for objective 2', 
                  accuracy=mean(pred==test$y), 
                  sensitivity=sensitivity(test$y, pred), 
                  specificity=specificity(test$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()


#Final model and evaluation against the validation set

#Now that we have our final models chosen, let's run the final evaluation against the $validation$ dataset.
pred1 <- ifelse(predict(obj1_top_1_model, newdata=validation[,-'duration'])==1,1,0)
pred2 <- ifelse(predict(obj1_top_2_model, newdata=validation[,-'duration'])==1,1,0)
pred <- ifelse(pred1==1 | pred2==1, 1, 0)

res <- data.frame(model='Final Validation Evaluation - objective 1', 
                  accuracy=mean(pred==validation$y), 
                  sensitivity=sensitivity(validation$y, pred), 
                  specificity=specificity(validation$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()

pred1 <- ifelse(predict(obj2_top_1_model, newdata=validation)==1,1,0)
pred2 <- ifelse(predict(obj2_top_2_model, newdata=validation)==1,1,0)
pred <- ifelse(pred1==1 | pred2==1, 1, 0)

res <- data.frame(model='Final Validation Evaluation - objective 2', 
                  accuracy=mean(pred==validation$y), 
                  sensitivity=sensitivity(validation$y, pred), 
                  specificity=specificity(validation$y, pred)) %>% 
  mutate(balanced_accuracy=(sensitivity+specificity)/2)
model_results <- bind_rows(model_results, res)
res %>% knitr::kable()

#Final summary of all the models trained along with the final evaluation on the validation set.
model_results %>% knitr::kable()





#Let's try a random forest using Rborist now. However, this will be very slow due to the size of the dataset.
#Instead, let's use PCA to get the 2-3 principal component vectors and train the random forest model on that.
#We will pick only the numerical features for this and try.
#cat_features <- c('contact', 'month', 'day_of_week', 'poutcome')
#train_pca <- dummy.data.frame(select(train, contact, month, day_of_week, campaign, poutcome, emp.var.rate, euribor3m, nr.employed, cons.price.idx, cons.conf.idx), names=cat_features)
#pca <- prcomp(train_pca, scale.=T)
pca <- prcomp(select(train, campaign, emp.var.rate, euribor3m, nr.employed, cons.price.idx, cons.conf.idx), scale.=T)
summary(pca)
train_reduced <- pca$x[,1:3]
rf_model <- train(x=train_reduced, y=train$y, method='rf')
test_reduced <- select(test, campaign, emp.var.rate, euribor3m, nr.employed, cons.price.idx, cons.conf.idx)
test_reduced <- sweep(as.matrix(test_reduced), 2, colMeans(as.matrix(test_reduced))) %*% pca$rotation
test_reduced <- test_reduced[,1:3]

#rf_model <- train(y~contact+month+day_of_week+campaign+poutcome+emp.var.rate+euribor3m+nr.employed+cons.price.idx+cons.conf.idx, data=train, method='Rborist')
pred <- predict(rf_model, newdata=test_reduced)
pca_rf_acc <- mean(pred==test$y)
pca_rf_acc#0.8872556
(caret::confusionMatrix(data=pred, reference=test$y))$byClass[c('Sensitivity', 'Specificity', 'Balanced Accuracy')]

rf_model <- train(y~contact+month+day_of_week, data=train, method='rf')
pred <- predict(rf_model, newdata=test)
rf_acc <- mean(pred==test$y)
rf_acc
(caret::confusionMatrix(data=pred, reference=test$y))$byClass[c('Sensitivity', 'Specificity', 'Balanced Accuracy')]
