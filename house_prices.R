####
# Packages 
#### 
library(tidyverse)
library(caret)
library(recipes)
library(Metrics)
library(ggcorrplot)
library(gridExtra)
library(missForest)
library(glmnet)
library(xgboost)
library(tensorflow)
library(keras)
library(patchwork)


####
# Import data 
#### 
train_data <- read.csv('data/input/train.csv')
test_data <- read.csv('data/input/test.csv') 

head(train_data)
head(test_data)

dim(train_data)
dim(test_data)


####
# Data Processing
#### 
# Target variable 
summary(train_data$SalePrice) 

plt1_target <- ggplot(train_data, aes(x = SalePrice)) + 
  geom_histogram(aes(y = ..density..), colour = 'black', fill = 'white') +
  geom_density(alpha = .2, fill = 'tomato') + 
  theme_bw()

plt2_target <- ggplot(train_data, aes(sample = SalePrice)) + 
  stat_qq() +
  stat_qq_line() + 
  labs(x = "", y = "") + theme_bw()

plt1_target | plt2_target 

## Log transformation
train_data <- train_data %>%
  mutate(
    SalePrice = log(SalePrice)
  )

plt3_target <- ggplot(train_data, aes(x = SalePrice)) + 
  geom_histogram(aes(y = ..density..), colour = 'black', fill = 'white') +
  geom_density(alpha = .2, fill = 'tomato') + 
  theme_bw()

plt4_target <- ggplot(train_data, aes(sample = SalePrice)) + 
  stat_qq() +
  stat_qq_line() + 
  labs(x = "", y = "") + theme_bw()

plt3_target | plt4_target 


# Outliers 
numerical_vars <- sapply(train_data, is.numeric)

corr_mat <- round(cor(train_data[, numerical_vars]), 2)

ggcorrplot(corr_mat, lab = F, ggtheme = ggplot2::theme_bw(),
           type = 'lower', colors = c("#6D9EC1", "white", "#E46726"))


most_corr_mat <- corr_mat[abs(corr_mat['SalePrice', ]) >= .5, abs(corr_mat['SalePrice', ]) >= .5]

ggcorrplot(most_corr_mat, lab = T, ggtheme = ggplot2::theme_bw(),
           type = 'lower', colors = c("#6D9EC1", "white", "#E46726"),
           lab_size = 3)


plt1 <- train_data %>%
  ggplot(aes(x = OverallQual, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt2 <- train_data %>%
  ggplot(aes(x = GrLivArea, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt3 <- train_data %>%
  ggplot(aes(x = GarageCars, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt4 <- train_data %>%
  ggplot(aes(x = GarageArea, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt5 <- train_data %>%
  ggplot(aes(x = TotalBsmtSF, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt6 <- train_data %>%
  ggplot(aes(x = X1stFlrSF, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt7 <- train_data %>%
  ggplot(aes(x = YearBuilt, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt8 <- train_data %>%
  ggplot(aes(x = YearRemodAdd, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt9 <- train_data %>%
  ggplot(aes(x = FullBath, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

plt10 <- train_data %>%
  ggplot(aes(x = TotRmsAbvGrd, y = SalePrice)) + 
  geom_point() + 
  geom_smooth(se = F) + 
  theme_bw()

grid.arrange(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8,
             plt9, plt10)

## Remove outlier 
train_data[which(train_data$TotalBsmtSF > 6000), ]
train_data <- train_data[-which(train_data$TotalBsmtSF > 6000), ]


####
# Feature engineering
#### 
# Merge datasets
n_train <- nrow(train_data)
n_test <- nrow(test_data)

train_data_target <- train_data$SalePrice

all_data <- rbind(train_data[, - ncol(train_data)], test_data)
dim(all_data)

# Missing data 
all_data_long <- all_data %>%
  gather(key = "variable", value = "valor", -Id)

all_data_long %>%
  group_by(variable) %>% 
  dplyr::summarize(porcentaje_NA = 100 * sum(is.na(valor)) / length(valor)) %>%
  filter(porcentaje_NA > 0) %>%
  ggplot(aes(x = reorder(variable, desc(porcentaje_NA)), y = porcentaje_NA)) +
  geom_col() +
  labs(title = "% missing data by feature",
       x = "Feature", y = "% missing data") +
  theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

## Impute missing values 
all_data <- all_data %>%
  mutate(
    Alley = replace(Alley, is.na(Alley), "None"),
    BsmtQual = replace(BsmtQual, is.na(BsmtQual), "None"),
    BsmtCond = replace(BsmtCond, is.na(BsmtCond), "None"),
    BsmtExposure = replace(BsmtExposure, is.na(BsmtExposure), "None"),
    BsmtFinType1 = replace(BsmtFinType1, is.na(BsmtFinType1), "None"),
    BsmtFinType2 = replace(BsmtFinType2, is.na(BsmtFinType2), "None"),
    FireplaceQu = replace(FireplaceQu, is.na(FireplaceQu), "None"),
    GarageType = replace(GarageType, is.na(GarageType), "None"),
    GarageFinish = replace(GarageFinish, is.na(GarageFinish), "None"),
    GarageQual = replace(GarageQual, is.na(GarageQual), "None"),
    GarageCond = replace(GarageCond, is.na(GarageCond), "None"),
    PoolQC = replace(PoolQC, is.na(PoolQC), "None"),
    Fence = replace(Fence, is.na(Fence), "None"),
    MiscFeature = replace(MiscFeature, is.na(MiscFeature), "None"),
    GarageYrBlt = replace(GarageYrBlt, is.na(GarageYrBlt), 0),
    GarageArea = replace(GarageArea, is.na(GarageArea), 0),
    GarageCars = replace(GarageCars, is.na(GarageCars), 0),
    BsmtFinSF1 = replace(BsmtFinSF1, is.na(BsmtFinSF1), 0),
    BsmtFinSF2 = replace(BsmtFinSF2, is.na(BsmtFinSF2), 0),
    BsmtUnfSF = replace(BsmtUnfSF, is.na(BsmtUnfSF), 0),
    TotalBsmtSF = replace(TotalBsmtSF, is.na(TotalBsmtSF), 0),
    BsmtFullBath = replace(BsmtFullBath, is.na(BsmtFullBath), 0),
    BsmtHalfBath = replace(BsmtHalfBath, is.na(BsmtHalfBath), 0),
    MasVnrType = replace(MasVnrType, is.na(MasVnrType), "None"),
    MasVnrArea = replace(MasVnrArea, is.na(MasVnrArea), 0),
    Functional = replace(Functional, is.na(Functional), "Typ"),
    MSSubClass = replace(MSSubClass, is.na(MSSubClass), "None"),
  )

all_data[sapply(all_data, is.character)] <- lapply(all_data[sapply(all_data, is.character)], as.factor)

all_data_imp <- missForest(as.data.frame(all_data), verbose = F)
all_data <- all_data_imp$ximp

all_data_long <- all_data %>%
  gather(key = "variable", value = "valor", -Id)

all_data_long %>%
  group_by(variable) %>% 
  dplyr::summarize(porcentaje_NA = 100 * sum(is.na(valor)) / length(valor)) %>%
  filter(porcentaje_NA > 0) %>%
  ggplot(aes(x = reorder(variable, desc(porcentaje_NA)), y = porcentaje_NA)) +
  geom_col() +
  labs(title = "% missing data by feature",
       x = "Feature", y = "% missing data") +
  theme_bw() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Categorical data 
all_data <- all_data %>%
  mutate(
    OverallCond = as.factor(OverallCond),
    YrSold = as.factor(YrSold),
    MoSold = as.factor(MoSold)
  )
glimpse(all_data)

# Add one variable 
all_data <- all_data %>%
  mutate(
    TotalSF = TotalBsmtSF + X1stFlrSF + X2ndFlrSF
  )


# Recipes 
train_data <- all_data[1:n_train, ]
train_data$SalePrice <- train_data_target
test_data <- all_data[(n_train + 1):nrow(all_data), ]

objeto_recipe <- recipe(formula = SalePrice ~ .,
                        data = train_data)

objeto_recipe <- objeto_recipe %>%
  step_rm(Id) %>%
  step_corr(all_numeric()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes()) 

objeto_recipe

trained_recipe <- prep(objeto_recipe, training = train_data)
trained_recipe

train_data_prep <- bake(trained_recipe, new_data = train_data)
test_data_prep <- bake(trained_recipe, new_data = test_data)

write.csv(train_data_prep, 'data/output/train_prep.csv', row.names = F)
write.csv(test_data_prep, 'data/output/test_prep.csv', row.names = F)

#train_data_prep <- read.csv('data/output/train_prep.csv')
#test_data_prep <- read.csv('data/output/test_prep.csv')

####
# Random Forest 
#### 
house_trControl <- trainControl(
  method = 'cv',
  number = 15, 
  savePredictions = 'final'
)

set.seed(1234)
house_rf <- caret::train(
  SalePrice ~ .,
  data = train_data_prep,
  method = 'rf',
  metric = 'RMSE',
  tuneGrid = expand.grid(mtry = seq(from = 26, to = 50, by = 2)),
  trControl = house_trControl
)
house_rf

plot(house_rf)

rmsle(actual = house_rf$pred[, 'obs'],
      predicted = house_rf$pred[, 'pred'])

rmse_rf <- RMSE(pred = house_rf$pred[, 'pred'],
                obs = house_rf$pred[, 'obs']) 
rmse_rf


####
# LASSO regression 
#### 
train_data_X <- model.matrix(SalePrice ~ -1 + ., data = train_data_prep)
train_data_Y <- train_data_prep$SalePrice

cv.lasso <- cv.glmnet(train_data_X, train_data_Y, alpha = 1)
plot(cv.lasso)

house_lasso <- glmnet(train_data_X, train_data_Y, alpha = 1, lambda = cv.lasso$lambda.min) 
house_lasso_coefs <- as.matrix(coef(house_lasso))[,1]
house_lasso_coefs

house_lasso_pred <- predict(house_lasso, newx = train_data_X)
rmsle_lasso <- RMSE(pred = house_lasso_pred,
                obs = train_data_Y) 
rmsle_lasso


####
# Neural Network 
#### 
train_data_nn <- train_data_X
train_data_labels <- as.matrix(train_data_Y)
paste0("Training entries: ", nrow(train_data_nn), ", labels: ", nrow(train_data_labels))

build_model <- function()
{
  
  model <- keras_model_sequential() %>%
    layer_dense(input_shape = dim(train_data_nn)[2], units = 32, activation = 'relu') %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'relu')
  
  model %>% compile(loss = 'mse',
                    optimizer = optimizer_rmsprop(),
                    metrics = 'mean_squared_error')
  
  model
  
}
house_nn <- build_model()

house_nn %>% summary()

epochs <- 500
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 80)
history <- house_nn %>% 
  fit(
    train_data_nn,
    train_data_labels,
    epochs = epochs,
    validation_split = 0.25,
    callbacks = list(early_stop)
  )

plot(history, metrics = 'mean_squared_error', smooth = F) +
  coord_cartesian(ylim = c(0, 1)) + 
  theme_bw()

house_pred_nn <- house_nn %>% predict(train_data_X)
rmsle_nn <- RMSE(pred = house_pred_nn,
                 obs = train_data_prep$SalePrice)
rmsle_nn


####
# Prediction and submission
####  
test_data_X <- as.matrix(test_data_prep)
pred <- as.vector(exp(predict(house_lasso, test_data_X)))
head(pred)

df_submission <- data.frame(Id = test_data$Id,
                            SalePrice = pred)
head(df_submission)

write.csv(df_submission, 'data/output/submission.csv', row.names = F)
