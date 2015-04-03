setwd("/Users/okada/myWork/kaggle/otto/")
trainingall <- read.csv("train.csv")
testing  <- read.csv("test.csv")

# 93 features, 9 classes.
# The objective is to build a model that can classifly Class1 - Class9 from 93 features
head(trainingall)
summary(trainingall)
# we don't have any missing values
any(is.na(trainingall)) 
any(is.na(testing)) 

## basic random forest
library(caret)
library(doMC)
registerDoMC(cores=4)
set.seed(39104)
inTrain <- createDataPartition(y=trainingall$target, p=0.8, list=FALSE)
# remove ID column in the first column
training <- trainingall[inTrain, -1];cv <- trainingall[-inTrain, -1]
dim(training);dim(cv)
control <- trainControl (method = "cv", number = 5, repeats = 1)

###################
## Random Forest ##
###################
before <- proc.time()
modFitRF <- train(target~., data=training, method="rf", trControl = control)
proc.time() - before

predrf <- predict(modFitRF, cv)
confusionMatrix(predrf, cv$target)

plot(varImp (modFitRF, scale = FALSE), top = 20)
trellis.par.set(caretTheme())
plot (modFitRF, type = c("g", "o"))
save(modFitRF, file="modFitRF.RData")

predrf.test <- predict(modFitRF, testing)
cls1 <- ifelse(predrf.test=="Class_1", 1, 0)
cls2 <- ifelse(predrf.test=="Class_2", 1, 0)
cls3 <- ifelse(predrf.test=="Class_3", 1, 0)
cls4 <- ifelse(predrf.test=="Class_4", 1, 0)
cls5 <- ifelse(predrf.test=="Class_5", 1, 0)
cls6 <- ifelse(predrf.test=="Class_6", 1, 0)
cls7 <- ifelse(predrf.test=="Class_7", 1, 0)
cls8 <- ifelse(predrf.test=="Class_8", 1, 0)
cls9 <- ifelse(predrf.test=="Class_9", 1, 0)
submission.rf <- data.frame(id=testing$id, 
                            Class_1=cls1, Class_2=cls2, Class_3=cls3,
                            Class_4=cls4, Class_5=cls5, Class_6=cls6,
                            Class_7=cls7, Class_8=cls8, Class_9=cls9
)
write.csv(submission.rf, file="submission_rf.csv", row.names=FALSE, quote=FALSE)


#########
## GBM ##
#########
modFitGBM <- train(target~., data=training, method="gbm",trControl = control)
print(modFitGBM)
summary(modFitGBM$finalModel)
predGBM <- predict(modFitGBM, cv)
confusionMatrix(predGBM, cv$classe)
# which model parameters were most effective and ultimately selected for the final mode.
trellis.par.set (caretTheme())
plot (modFitGBM, type = c("g", "o"))

plot(varImp (modFitGBM, scale = FALSE), top = 20)

###########
## LASSO ##
###########
library(caret)
library(doMC)
registerDoMC(cores=4)
before <- proc.time()
modFitSVMLn <- train(target~., data=training, method="lasso", trControl = control)
proc.time() - before
predLasso <- predict(modFitLasso, cv)
confusionMatrix(predLasso, cv$target)
save(modFitLasso, file="modFitLasso.RData")

###########
## SVM ##
###########
library(caret)
library(doMC)
registerDoMC(cores=4)
before <- proc.time()
modFitSVMLn <- train(target~., data=training, method="svmLinear", trControl = control)
proc.time() - before

predsvmln <- predict(modFitSVMLn, cv)
confusionMatrix(predsvmln, cv$target)
save(modFitSVMLn, file="modFitSVMLn.RData")

