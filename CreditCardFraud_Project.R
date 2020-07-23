# ---
# title: "Credit Card Fraud Detection"
# author: "Rishabh Singh Dodeja"
# date: "July 17, 2020"
# ---
# This R Script will perform the following task:
#   1. Download the kaggle creditcardfraud dataset and required libraries (if needed)
#   2. Clean, Normalize & Split the data set into train(80%) and test(20%) 
#   3. The data after split resampled using SMOTE and K-Fold Cross Validation (to avoid underfitting and overfitting due to biased dataset)
#   4. Build and train a Naive Random Forest Model then evaluate results on 
#   5. Build, train and test a more optimized Random Forest model achieved after variable filtering and threshold tunind (explained in Report/Rmd)
#   6. Build, train and test a XG-Boost Classifier Model
#   7. Generate Confusion Matrix and ROC Plots with AUC for each model
#   8. Generate final evaluation table with all the model and evaluation scores in following metrics
#       "Specificty", "F1-Score" and "AUC"
#
# Let's Begin!
#
################################
# Create creditcard dataset
################################
#
# Note: this process could take a couple of minutes
#
# Load/install required library/packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(Mlmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
#
#Credit Card Fraud Data
#Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud/
#
# Direct download and read data
dl <- tempfile()
download.file("https://www.kaggle.com/mlg-ulb/creditcardfraud/download", dl)
#
creditcard <- fread(text = gsub("::", "\t", readLines(unzip(dl, "creditcard.csv"))))
#
#Read data from local directory
#creditcard <- fread(text = gsub("::", "\t", readLines("creditcard.csv")))
#
####################################
# Prepare Data for training/testing
####################################
#
## Normalization
#
# function to normalize columns/arrays
normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
}
#
#All Columns/Variables V1-V28 are already normalized
#
# Normalize Amount variable before putting into model.  
creditcard$Amount <- normalize(creditcard$Amount)
#
## Split Train/Test
#
set.seed(56)
#
#Create Data Partition
train_index = createDataPartition(creditcard$Class, times = 1, p = 0.8, list = F)
#
#Distributing data to test and train sets
train = creditcard[train_index]
test = creditcard[!train_index]
train$Class <- as.factor(train$Class)
test$Class <- as.factor(test$Class)
levels(train$Class)=make.names(c("Genuine","Fraud"))
levels(test$Class)=make.names(c("Genuine","Fraud"))
#
# Uncomment registerDoMC to activate parallel processing
# Parallel processing for faster training
#registerDoMC(cores = 4)
#
#
## K-Fold Cross Vallidation and SMOTE
#
# Use 10-fold cross-validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",
                     summaryFunction = twoClassSummary,
                     savePredictions = T)
#
#
####################################
# Random Forest Classifier [Naive]
####################################
#
##install caret, Classification Regresiion and training package
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
#
#train RendomForst Model
RF_Model <- train(Class ~ ., data = train, method = "rf", trControl = ctrl, verbose = T, metric = "ROC")
#
#get prediction for test
preds = predict(RF_Model, test, type = "prob")
#
# threshold is initially selected as 0.5
pred = as.factor(preds$Fraud>0.5)
levels(pred)=make.names(c("Genuine","Fraud"))
y_test = test$Class
#
#Calcuate Confusion Matrix
conf_mat_RF <- confusionMatrix(pred,y_test)
#
#Plot Confusion matrix table as four-fould plor
fourfoldplot(conf_mat_RF$table)
#
#Evaluate AUC and Plot ROC
roc_data <- roc(y_test, predict(RF_Model, test, type = "prob")$Fraud)
plot(roc_data, main = paste0("AUC: ", round(pROC::auc(roc_data), 5)))
#
# Calculate Evluation Table
#Specificty
Sp_RF = as.numeric(conf_mat_RF$byClass["Specificity"])

#RF F1_Score
F1_RF = round(F1_Score(y_test,pred),5)

#AUC
roc_data = roc(y_test, predict(RF_Model, test, type = "prob")$Fraud)
AUC_RF = round(pROC::auc(roc_data), 5)

# Create Results Table
result = tibble(Method = "Random Forest", Specificty = Sp_RF , F1Score = F1_RF, AUC = AUC_RF)
#
#
#######################################
# Random Forest Classifier [Optimized]
#######################################
# 
# Now, we can build our final Random Forest Model with top 10 most inmportant variables 
# Variables = V14+V10+V17+V4+V12+V16+V11+V3+V7+V2
# Threshhold = 0.5
# For analysis about the values above check analysis in Rmd/Report File
#
# train 10-variable model
RF_Model = train(Class ~ V14+V10+V17+V4+V12+V16+V11+V3+V7+V2, data = train, method = "rf", trControl = ctrl, verbose = T, metric = "ROC")
#
# predict test dataset
preds <- predict(RF_Model, test, type = "prob")
#
# threshold is selected as 0.5
pred = as.factor(preds$Fraud>0.5)
levels(pred)=make.names(c("Genuine","Fraud"))
#
# Confusion Matrix 
conf_mat_RF <- confusionMatrix(pred,y_test)
#
# Confusion Matrix Plot
fourfoldplot(conf_mat_RF$table)
#
# plot ROC
roc_data = roc(y_test, predict(RF_Model, test, type = "prob")$Fraud)
plot(roc_data, main = paste0("AUC: ", round(pROC::auc(roc_data), 5)))
#
#Add to final Evaluation Table
#
#Specificty
Sp_RF = as.numeric(conf_mat_RF$byClass["Specificity"])
#
#RF F1_Score
F1_RF = round(F1_Score(y_test,pred),5)
#
#AUC
roc_data = roc(y_test, predict(RF_Model, test, type = "prob")$Fraud)
AUC_RF = round(pROC::auc(roc_data), 5)
#
# Add to Evaluation Table
result = bind_rows(result, tibble(Method = "Random Forest (Optimized)", Specificty = Sp_RF , F1Score = F1_RF, AUC = AUC_RF))
#
#
######################
# XG-Boost Classifier
######################
#
#Installing and loading Xgboost Package
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
#
#recreating test/train dataset for xgb based on previos train_index value 
#as we tranformed Class coloum to factor in previous, it creats problem with XGB
train = creditcard[train_index]
test = creditcard[!train_index]
#
#create Data Matrix form for XGB
dtrain <- xgb.DMatrix(data = as.matrix(train[,-c("Class")]), label = train$Class)
dtest <- xgb.DMatrix(data = as.matrix(test[,-c("Class")]), label = test$Class)
#
#Build and train model
xgb <- xgboost(data = dtrain, nrounds = 100, gamma = 0.1, max_depth = 10, objective = "binary:logistic", nthread = 7)
#
#
#Run Predictions
preds_xgb <- predict(xgb, dtest)
#
#Convert Prediction to Factors for Confusion Matrix
pred_fac = as.factor(preds_xgb>0.5)
levels(pred_fac)= make.names(c("Genuine","Fraud"))
y_test = as.factor(test$Class)
levels(y_test)=make.names(c("Genuine","Fraud"))
#
#
# Confusion Matrix
conf_mat_XGB = confusionMatrix(pred_fac, y_test)
#
# Confusion Matrix Four-Fold Plot
fourfoldplot(conf_mat_XGB$table)
#
#plot ROC
roc_data <- roc(y_test, preds_xgb)
plot(roc_data, main = paste0("AUC: ", round(pROC::auc(roc_data), 5)))
#
#
# Complete Evaluation Table
Sp_XGB = as.numeric(conf_mat_XGB$byClass["Specificity"])
#
#RF F1_Score
F1_XGB = round(F1_Score(y_test,pred_fac),5)
#
#AUC
roc_data = roc(y_test, preds_xgb)
AUC_XGB = round(pROC::auc(roc_data), 5)
#
# Create Results Table
result = bind_rows(result, tibble(Method = "XG-Boost", Specificty = Sp_XGB , F1Score = F1_XGB, AUC = AUC_XGB))
#
##########
# Results
##########
# Print result table
print(result)