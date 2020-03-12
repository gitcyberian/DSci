# NOTE: This script first downloads the datasets using the script shared in the assignment instructions,
#so an internet connection is required and the download may take a few minutes.
#After the datasets are downloaded, the rest of the script takes about 2 minutes to run on an 8 GB RAM machine.
#The script also uses a few packages that may not already be installed.
#The script checks for the package availabliity and installs them if they are not present.
#Specifically, the anytime package is used by this script - this is not a package referenced in the course material
#Hence, this package may not already be installed on your system in which case the script will install the package.

#Load the required libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(dplyr)
library(anytime)
library(lubridate)
library(caret)
library(data.table)

################################
# Create edx set, validation set
################################

# Note: this process could take a few minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#Download the data sets from the movielens site.
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#Remove the variables that we don't need to clear memory.
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#################################################
# Wrangle the data to get the year and genres out
#################################################
#Let's get the year of movie release into a separate column. Use regex to identify the (year) within the title
temp <- str_match(edx$title, '(.*)[(](\\d{4})[)]$')[,2:3]#extracts 2 groups - the title in the 1st and the year in the 2nd.
edx$title <- str_trim(temp[,1])#trim the trailing whitespace in the extracted title
edx$year <- as.integer(temp[,2])
rm(temp) #remove variable to free up memory

#Extract the individual genres from the genres column in edx.
genres <- data.frame(unique(edx$genres), stringsAsFactors = FALSE)
names(genres) <- 'genregroup'
#We can use the | separator to separate out the individual genres from the genres column
#Then get the unique ones out to get the list of all individual genres
genres <- unique(genres %>% separate(genregroup, into=paste('genre', 1:9, sep=''), fill="right", sep='[|]') %>%
                   gather(genre, value, paste('genre', 1:9, sep='')) %>% filter(!is.na(value) & value!='(no genres listed)') %>% pull(value))

#We now have 19 individual genres that each film may or may not be tagged to.
#Let's create a mapping dataframe called movie_genres that maps the movies to the individual genres it belongs to.
movie_genres <- 
  edx %>% select(movieId, genres) %>% group_by(movieId, genres) %>% sample_n(1) %>% ungroup() %>%
  separate(genres, into=paste('genre', 1:9, sep=''), fill="right", sep='[|]') %>%
  gather(key, genre, paste('genre', 1:9, sep='')) %>% filter(!is.na(genre)) %>%
  select(movieId, genre) %>% group_by(movieId) %>% mutate(genre_count=n())

#We don't need the genres column anymore in edx as we have extracted the individual genres and created the mapping.
edx <- edx %>% select(-genres)

#################################
# Generating the prediction model
#################################

#Function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Generate the model now.
#Our model will be:
#predicted_rating = mean + movie bias factor + user bias factor + date bias factor + genre bias factor
#Mean rating for the entire dataset
mu <- edx %>% pull(rating) %>% mean()

#Regularisation factor set to 5 based on iterative analysis and testing
#Refer to the rmd or report for more details on how this factor was selected.
l <- 5

#Generate the movie bias factor values.
#We can do this by calculating average(rating - mean) grouped by movie.
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

#Generate the user bias factor values.
#We can do this by calculating average(rating - mean - movie bias) grouped by user.
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#Generate the date bias factor values (the date bias is based on the number of years between the movie release and the date of user rating)
#Refer to the rmd file or the pdf report for a more detailed explanation of why this was done.
#We can do this by calculating average(rating - mean - movie bias - user bias) grouped by review age.
b_d <- edx %>% select(userId,movieId,rating,timestamp,year) %>% 
  mutate(date_diff=round((year(anydate(timestamp))-year))) %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(date_diff) %>%
  summarize(b_d = sum(rating - mu - b_i - b_u)/(n()+l))

#Generate the genre bias factor values.
#We can do this by calculating average(rating - mean - movie bias - user bias - date bias) grouped by genre.
#Note that a movie can belong to multiple genres, hence divide by the genre count for individual rating records while grouping by genre
genre_avgs <- edx %>% select(-title) %>% left_join(movie_genres, by='movieId') %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(date_diff=round(year(anydate(timestamp))-year)) %>% 
  left_join(b_d, by='date_diff') %>%
  mutate(b_g_ui = (rating - mu - b_i - b_u - b_d)/genre_count) %>%
  group_by(genre) %>% summarize(b_g=mean(b_g_ui))

#Model is ready now. We can now generate predictions for the validation set.

###################################################
# Generating the predictions for the validation set
###################################################
#Let's first get the year of movie release into a separate column for the validation set
temp <- str_match(validation$title, '(.*)[(](\\d{4})[)]$')[,2:3]#extracts 2 groups - the title in the 1st and the year in the 2nd.
validation$title <- str_trim(temp[,1])#trim the trailing whitespace in the extracted title
validation$year <- as.integer(temp[,2])
rm(temp) #remove variable to free up memory

#We are now ready to generate the predictions.
#predicted_rating = mean + movie bias factor + user bias factor + date bias factor + genre bias factor (for all genres for that particular movie)
predicted_ratings <- validation %>% select(-c(title, genres)) %>% left_join(movie_genres, by='movieId') %>% 
  left_join(genre_avgs, by='genre') %>%
  group_by(movieId, userId) %>% mutate(sum_b_g=sum(b_g)) %>% sample_n(1) %>% ungroup() %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(date_diff=round(year(anydate(timestamp))-year)) %>% 
  left_join(b_d, by='date_diff') %>%
  mutate(pred = mu + b_i + b_u + b_d + sum_b_g) %>%
  select(movieId, userId, pred, rating)

#Set the predicted rating to just the mean for any movies/users in validation set that were not there in the edx dataset
predicted_ratings[is.na(predicted_ratings$pred), 'pred'] <- mu
#Cut-offs for predicted rating should be 0.5 and 5.
predicted_ratings[predicted_ratings$pred<0, 'pred'] <- 0.5 #if there are any -ve predictions
predicted_ratings[predicted_ratings$pred>5, 'pred'] <- 5 #if there any predictions > 5

#############################
# RMSE for the validation set
#############################
#Calculate the RMSE error for the validation set
rmse_val <- RMSE(predicted_ratings$pred, predicted_ratings$rating)
print(rmse_val)
