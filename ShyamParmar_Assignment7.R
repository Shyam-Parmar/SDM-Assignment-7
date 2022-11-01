################################
# Assignment 7

################################
# Additional Code

# function that converts 'No' to 0 and 'Yes' to 1
#value_converter <- function(x) { ifelse (x == 'No', 0, 1 ) }

# Apply value converter across the identified columns
#df <- df %>% mutate(across(all_of(binary_cols), value_converter))

################################

# Load Libraries
pacman::p_load(dplyr, tidyr, caret, ggplot2, caTools, MLmetrics, mlbench, mlTools, corrplot, expss, PerformanceAnalytics, AER, MASS, stargazer, pscl, jtools, Hmisc, ggcorrplot, rpart, rpart.plot, readxl, ROCR)

# Clear environment
rm(list=ls())

# Import Dataset
df <- read_excel('C:/Users/Scott/Downloads/TelcoChurn.xlsx', sheet = 'Data')

###############################
# Clean the dataset

# Drop NA Values if exist
colSums(is.na(df))
df <- df[complete.cases(df), ]

# Convert char to binary #####

# Create new column to simplify whether customer has internet service or not 
df$IntService <- ifelse(df$InternetService == "No", "No", "Yes")

df <- subset(df, select = -c(customerID, InternetService))

# Identify binary columns
binary_cols <- c('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'IntService')

# Categorical columns 1
cat_cols_one <- c('OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies')

# Categorical columns 2
cat_cols_two <- c('MultipleLines')

# convert columns to factor and levels to No
for (i in c(binary_cols, cat_cols_one, cat_cols_two)) {
  df[[i]] <- relevel(factor(df[[i]]), ref = "No")
}

# Factorize the rest of the data
df$gender <- factor(df$gender, levels = c("Male", "Female"))
df$Contract <- factor(df$Contract,levels = c("Month-to-month", "One year", "Two year"), ordered = TRUE)
df$PaymentMethod <- factor(df$PaymentMethod, levels = c("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
df$SeniorCitizen <- factor(df$SeniorCitizen, levels = c("0", "1"))

df$Churn <- factor(ifelse(df$Churn == "No", 0, 1), levels = c("0", "1"))
levels(df$Churn)

# Exploratory Analysis
table(df$PhoneService)
table(df$IntService)
cross_cases(df, PhoneService, IntService) %>% if_na(0)

table(df$Churn)
cross_cases(df, Churn, PhoneService) %>% if_na(0)
cross_cases(df, Churn, IntService) %>% if_na(0)

###############################
# Model 1 Only Phone Services
# Partition the dataframe
df_phone <- subset(df, df$PhoneService == "Yes" & df$IntService == "No")

# Set Seed and Split the dataframe
set.seed(1024)
trainIndex <- sample(1:nrow(df_phone), size=round(0.75*nrow(df_phone)), replace=FALSE)
train <- df_phone[trainIndex,]
test  <- df_phone[-trainIndex,]
dim(train); dim(test)

str(df_phone)

# Model
phone <- glm(Churn ~ gender + SeniorCitizen + Partner + Dependents + MultipleLines + tenure + Contract + PaperlessBilling + PaymentMethod + TotalCharges, data=train, family=binomial (link="logit"))
summary(phone)

# Predict
predphone <- predict(phone, test, type="response")
predphone <- ifelse(predphone>0.5, 1, 0)

# COnfusion Matrix
confusionMatrix(data = factor(predphone, levels = c("0", "1")), reference = test$Churn)

# ROC and AUC
pr <- prediction(predphone, test$Churn)
prf <- performance(pr, measure="tpr", x.measure="fpr")
plot(prf)                                                 # ROC plot: TPR vs FPR

auc <- performance(pr, measure="auc")
auc <- auc@y.values[[1]]
auc 

recall_tel = Recall(test$Churn, predphone, positive = NULL)
print(recall_tel)
precision_tel = Precision(test$Churn, predphone, positive = NULL)
print(precision_tel)
f1_tel = F1_Score(predphone,test$Churn)
print(f1_tel)
auc_tel = AUC(predphone,test$Churn)
print(auc_tel)

##################################
# Model 2 Only Internet Service
df_internet <- subset(df, df$PhoneService == "No" & df$IntService == "Yes")

# Set Seed and Split the dataframe
set.seed(1024)
trainIndex <- sample(1:nrow(df_internet), size=round(0.75*nrow(df_internet)), replace=FALSE)
train <- df_internet[trainIndex,]
test  <- df_internet[-trainIndex,]
dim(train); dim(test)

str(df_internet)

# Model
internet  <- glm(Churn ~ gender + SeniorCitizen + Partner + Dependents + OnlineSecurity + OnlineBackup + 
                   DeviceProtection + TechSupport + StreamingTV + StreamingMovies + tenure + Contract + 
                   PaperlessBilling + PaymentMethod + TotalCharges, data=train, family=binomial (link="logit"))
summary(internet)

# Predict
predinternet <-predict(internet, test, type="response")
predinternet <- ifelse(predinternet>0.5, 1, 0)

# COnfusion Matrix
confusionMatrix(data = factor(predinternet, levels = c("0", "1")), reference = test$Churn)

# ROC and AUC
pr <- prediction(predinternet, test$Churn)
prf <- performance(pr, measure="tpr", x.measure="fpr")
plot(prf)                                                 # ROC plot: TPR vs FPR

auc <- performance(pr, measure="auc")
auc <- auc@y.values[[1]]
auc 

recall_tel = Recall(test$Churn, predinternet, positive = NULL)
print(recall_tel)
precision_tel = Precision(test$Churn, predinternet, positive = NULL)
print(precision_tel)
f1_tel = F1_Score(predinternet,test$Churn)
print(f1_tel)
auc_tel = AUC(predinternet,test$Churn)
print(auc_tel)

#################################
# Model 3 Both Phone and Internet
df_both <- subset(df, df$PhoneService == "Yes" & df$IntService == "Yes")

df_both <- subset(df_both, select = c(-PhoneService, -IntService))

# Set Seed and Split the dataframe
set.seed(1024)
trainIndex <- sample(1:nrow(df_both), size=round(0.75*nrow(df_both)), replace=FALSE)
train <- df_both[trainIndex,]
test  <- df_both[-trainIndex,]
dim(train); dim(test)

str(df_both)

# Model
both  <- glm(Churn ~ . , data=train, family=binomial (link="logit"))
summary(both)

# Predict
predboth <-predict(both, test, type="response")
predboth <- ifelse(predboth>0.5, 1, 0)

# COnfusion Matrix
confusionMatrix(data = factor(predboth, levels = c("0", "1")), reference = test$Churn)

# ROC and AUC
pr <- prediction(predboth, test$Churn)
prf <- performance(pr, measure="tpr", x.measure="fpr")
plot(prf)                                                 # ROC plot: TPR vs FPR

auc <- performance(pr, measure="auc")
auc <- auc@y.values[[1]]
auc 

recall_tel = Recall(test$Churn, predboth, positive = NULL)
print(recall_tel)
precision_tel = Precision(test$Churn, predboth, positive = NULL)
print(precision_tel)
f1_tel = F1_Score(predboth,test$Churn)
print(f1_tel)
auc_tel = AUC(predboth,test$Churn)
print(auc_tel)

###################################
# Stargazer
stargazer(phone, internet, both, title="Churn Rate", object.names = TRUE, type="text", single.row=TRUE, digits = 3, out = "table1.html")

###################################
# Phone marginal effect

# Marginal Effects
exp(m1$coef)
exp(m2$coef)
exp(m3$coef)

LogitScalar <- mean(dlogis(predict(phone, type="link")))  # Mean of density function of y
LogitScalar*coef(logit)

# Internet marginal effect
LogitScalar <- mean(dlogis(predict(internet, type="link")))  # Mean of density function of y
LogitScalar*coef(logit)

# Both marginal effect
LogitScalar <- mean(dlogis(predict(both, type="link")))  # Mean of density function of y
LogitScalar*coef(logit)


