setwd("/Users/okada/myWork/kaggle/otto/")
trainingall <- read.csv("train.csv")
testing  <- read.csv("test.csv")

# 93 features, 9 classes.
# The objective is to build a model that can classifly Class1 - Class9 from 93 features
head(trainingall)
summary(training)
# we don't have any missing values
any(is.na(trainingall)) 
any(is.na(testing)) 

## basic random forest
library(caret)
set.seed(39104)
inTrain <- createDataPartition(y=trainingall$target, p=0.8, list=FALSE)
# remove ID column in the first column
training <- trainingall[inTrain, -1];cv <- trainingall[-inTrain, -1]
dim(training);dim(cv)
control <- trainControl (method = "cv", number = 5, repeats = 1)

###################
## Random Forest ##
###################
modFitRF <- train(target~., data=training, method="rf", trControl = control)
predrf <- predict(modFitRF, cv)
confusionMatrix(predrf, cv$target)

plot(varImp (modFitRF, scale = FALSE), top = 20)
trellis.par.set(caretTheme())
plot (modFitRF, type = c("g", "o"))

#########
## GBM ##
#########
modFitGBM <- train(taret~., data=training, method="gbm",trControl = control)
print(modFitGBM)
summary(modFitGBM$finalModel)
predGBM <- predict(modFitGBM, cv)
confusionMatrix(predGBM, cv$classe)
# which model parameters were most effective and ultimately selected for the final mode.
trellis.par.set (caretTheme())
plot (modFitGBM, type = c("g", "o"))

plot(varImp (modFitGBM, scale = FALSE), top = 20)

