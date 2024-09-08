library(tidyverse)
library(AMR)
library(xgboost)

prev_courses <- sample(c(0, 1, 2, 3, 4),
                       size = 1e4,
                       replace = TRUE,
                       prob = c(0.75, 0.1, 0.075, 0.05, 0.025))
cip_mic <- molMIC::mic_range(start=4, min=0.008) |>
  rev() |>
  sample(1e4, T, prob = c(.05,
                                 .05,
                                 .05,
                                 .1,
                                 .1,
                                 .15,
                                 .1,
                                 .1,
                                 .05,
                                 .05))

cip_mic <- log2(cip_mic)
cip_mic <- cip_mic + (prev_courses * 1)
cip_mic <- molMIC::force_mic(2 ** cip_mic)

tibble(prev_courses, cip_mic) %>% 
  mutate(cip_mic = as.mic(cip_mic)) %>% 
  ggplot(aes(x = cip_mic, color = as.factor(prev_courses))) + geom_bar()

# sir
cip_comm <- rbinom(1e4,
                  1,
                  0.025)
cip_sir_acquired <- ifelse(prev_courses * 0.2 + rnorm(1e4) > 0.5,
                                1,
                                0)
cip_sir <- pmax(cip_comm, cip_sir_acquired)

boxplot(prev_courses ~ cip_sir)

split_ratio <- floor(0.8 * length(cip_sir))

train_x <- prev_courses[seq(split_ratio)]
test_x <- prev_courses[-seq(split_ratio)]
train_y <- cip_sir[seq(split_ratio)]
test_y <- cip_sir[-seq(split_ratio)]

dtrain <- xgboost::xgb.DMatrix(matrix(train_x), label = train_y)
dtest <- xgboost::xgb.DMatrix(matrix(test_x), label = test_y)

model <- xgb.train(data = dtrain, nrounds = 100,
                   watchlist = list(train = dtrain,
                                    eval = dtest),
                   params = list(objective = "binary:logistic"))

preds <- predict(model, dtest)

plot(test_y ~ preds)

# continuous
x <- rnorm(1e4) 
y <- x^3 * 2 + rnorm(1e4)
y <- subset(y, x > 0)
x <- subset(x, x > 0)

plot(y ~ x)

split_ratio <- floor(0.8 * length(x))

train_x <- x[seq(split_ratio)]
test_x <- x[-seq(split_ratio)]
train_y <- y[seq(split_ratio)]
test_y <- y[-seq(split_ratio)]

dtrain <- xgboost::xgb.DMatrix(matrix(train_x), label = train_y)
dtest <- xgboost::xgb.DMatrix(matrix(test_x), label = test_y)

model <- xgb.train(data = dtrain, nrounds = 100,
          watchlist = list(train = dtrain,
                       eval = dtest))

preds <- predict(model, dtest)

plot(test_y ~ preds)

tibble(test_x, preds, test_y) %>% 
  pivot_longer(cols = all_of(c("preds", "test_y"))) %>% 
  ggplot(aes(x = test_x, y = value, color = name)) +
  geom_point()
