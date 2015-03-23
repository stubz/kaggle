# setwd("/Users/okada/myWork/kaggle/driver/")

files <- list.files("./data/drivers/")
for(i in 1:length(files)){
  datafiles <- list.files(paste("./data/drivers/", files[i], "/", sep=""), pattern="csv")
}

N_driver <- length(files)
N_trip <- 200
N_feature <- 12
dat.all <- as.data.frame(matrix(rep(NA_real_, (N_feature+2)*N_trip*N_driver), ncol=(N_feature+2)))
feature_names <- c("driver_id","trip_id", "max_turn", "avg_turn", "median_turn", 
                   "total_turn","max_speed","avg_speed","median_speed",
                   "max_acceleration","min_acceleration","avg_acceleration", 
                   "median_acceleration","total_stop_sec") 
colnames(dat.all) <- feature_names

driver_id <- 1;i<-1
calcFeature <- function(driver_id, feature_names, N_trip=200){
  datafiles <- list.files(paste("./data/drivers/", driver_id, "/", sep=""), pattern="csv")
  feature <- as.data.frame(matrix(rep(NA_real_, 
                                      length(feature_names)*N_trip*1), 
                                  ncol=length(feature_names)))
  for(i in 1:N_trip){
    dat <- read.csv(paste("./data/drivers/",driver_id, "/", datafiles[i], sep=""), header=TRUE)
    tot.sec <- nrow(dat)
    dat1 <- dat[-tot.sec[1],]
    dat2 <- dat[-1,]
    v <- dat2-dat1
    v1 <- as.matrix(v[-dim(v)[1],])
    v2 <- as.matrix(v[-1,])
    # radius
    rad.vec <- acos(diag(v1%*%t(v2)) / (sqrt(diag(v1%*%t(v1)))*sqrt(diag(v2%*%t(v2))) ))
    # max turn
    max.turn <- max(rad.vec, na.rm=TRUE)
    # avg turn
    avg.turn <- mean(rad.vec, na.rm=TRUE)
    # median turn
    med.turn <- median(rad.vec, na.rm=TRUE)
    # total turn
    tot.turn <- sum(abs(rad.vec), na.rm=TRUE)
    
    ## Distance per second  =  speed m/s
    dist.vec <- sqrt(diag(v1%*%t(v1)))
    # max speed per second
    max.speed <- max(dist.vec, na.rm=TRUE)
    # avg speed per second
    avg.speed <- mean(dist.vec, na.rm=TRUE)
    # median speed per second
    med.speed <- median(dist.vec, na.rm=TRUE)
    
    ## Acceleration
    ## v = v0 + at => a = (v-v0)/t  m/s^2
    accl <- diff(dist.vec, lag=1)
    # max acceleration
    max.accl <- max(accl, na.rm=TRUE)
    # min acceleration
    min.accl <- min(accl, na.rm=TRUE)
    # avg acceleration
    avg.accl <- mean(accl, na.rm=TRUE)
    med.accl <- median(accl, na.rm=TRUE) 
    
    ## stopping time
    tot.stop.sec <- dim(v1[v1[,"x"]==0 & v1[,"y"]==0, ])[1]
    if(is.null(tot.stop.sec)) tot.stop.sec <- 0
    feature[i,] <- c(driver_id, sub(".csv","",datafiles[i]), max.turn,
                     avg.turn, med.turn, tot.turn, 
                     max.speed, avg.speed, med.speed, max.accl, 
                     min.accl, avg.accl, med.accl, tot.stop.sec) 
  }
  return( feature )
}

library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(4, type = "SOCK"))
before <- proc.time()
dat.all <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  #print(paste(Sys.time(), " | Loop ",i, " is done...",sep=""))
  calcFeature(driver_id = files[i], feature_names=feature_names, N_trip=N_trip )
}
colnames(dat.all) <- feature_names
proc.time() - before
save(dat.all, file="dat.all.RData")

#ユーザ   システム       経過
#357.648      1.072   4888.557
# takes approx. 1.5 hours

###############
## Model Fit ##
###############
load("dat.all.RData")

driver_IDs <- sort(unique(dat.all$driver_id))
N_driver <- length(driver_IDs)
N_trip <- 200

dat.all1 <- dat.all
for(j in 3:14){
  dat.all1[, j] <- as.double(dat.all[, j])
  # replace -Inf
  dat.all1[is.infinite(dat.all1[, j]), j] <- 0
  # replace NA
  dat.all1[is.na(dat.all1[, j]), j] <- 0
}
dat.all1$driver_trip <- with(dat.all1, paste(driver_id, trip_id, sep="_"))
save(dat.all1, file="dat.all1.RData")
load("dat.all1.RData")

#id <- driver_IDs[j]
makeClust <- function(id, N_trip=200){
  feature <- subset(dat.all1, driver_id==id)  
  feature.km <- kmeans(feature[,-c(1,2)],2)
  clust_count <- table(feature.km$cluster)
  majority_clust <- names(clust_count[rev(order(clust_count))])[1]
  tmp.prob <- ifelse(feature.km$cluster==majority_clust, 1, 0)
  tmp.trip <- paste(id, 1:N_trip, sep="_")
  
  tmp.dat <- data.frame(
      driver_id = rep(id, N_trip),
      trip_id = seq(1, N_trip),
      driver_trip = tmp.trip,
      prob = tmp.prob 
    )
  return(tmp.dat)
}
library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(2, type = "SOCK"))
before <- proc.time()
driver_prob <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  makeClust(id = driver_IDs[i], N_trip=N_trip )
}
proc.time() - before
save(driver_prob, file="driver_prob.RData")
load("driver_prob.RData")

############################################################
## probability estimate by logistic regression ###
library(randomForest)
load("dat.all1.RData")
load("driver_prob.RData")
dat.all2 <- merge(dat.all1, driver_prob[, c("driver_trip","prob")],
                  by="driver_trip", all.x=TRUE)
driver_IDs <- sort(unique(driver_prob$driver_id))
N_driver <- length(driver_IDs)
N_trip <- 200
id <- driver_IDs[1]

calcProb <- function(id, N_trip=200){
  #require(randomForest)
  tmp.driver <- subset(dat.all2, driver_id == id)
  tmp.true <- subset(dat.all2, driver_id == id & prob == 1)
  tmp.false <- subset(dat.all2, driver_id != id)
  tmp.false <- tmp.false[sample(1:dim(tmp.false)[1], N_trip*3),]
  tmp.dat <- rbind(tmp.true, tmp.false)
  tmp.dat$target <- ifelse(tmp.dat$driver_id == id, 1, 0)
  
  fit <- randomForest(factor(target)~max_turn+avg_turn+median_turn+total_turn+
               max_speed+avg_speed+median_speed+max_acceleration+min_acceleration+
               avg_acceleration+median_acceleration+total_stop_sec, 
             data = tmp.dat)
  pred.driver <- predict(fit, newdata=tmp.driver, type="prob")
  final.out <- data.frame(driver_trip=tmp.driver$driver_trip, prob=pred.driver[,2])
  return(final.out)
}

train_method<-"gbm"
train_method<-"rf"
calcProb2 <- function(id, train_method, N_trip=200){
  #require(randomForest)
  tmp.driver <- subset(dat.all2, driver_id == id)
  tmp.true <- subset(dat.all2, driver_id == id & prob == 1)
  tmp.false <- subset(dat.all2, driver_id != id)
  tmp.false <- tmp.false[sample(1:dim(tmp.false)[1], N_trip*3),]
  tmp.dat <- rbind(tmp.true, tmp.false)
  tmp.dat$target <- ifelse(tmp.dat$driver_id == id, "true", "false")  
  control <- trainControl (method = "cv", number = 3, repeats = 1)
  fit <- train(factor(target)~max_turn+avg_turn+median_turn+total_turn+
                 max_speed+avg_speed+median_speed+max_acceleration+min_acceleration+
                 avg_acceleration+median_acceleration+total_stop_sec, 
               data = tmp.dat, method=train_method, trControl = control)
  pred.driver <- predict(fit, newdata=tmp.driver, type="prob")
  final.out <- data.frame(driver_trip=tmp.driver$driver_trip, prob=pred.driver[,2])
  return(final.out)
}

library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(2, type = "SOCK"))
before <- proc.time()
#driver_prob_rf <- foreach(i=1:3, .combine="rbind") %dopar% {
driver_prob_rf <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  require(randomForest)
  calcProb(id = driver_IDs[i], N_trip=N_trip )
}
proc.time() - before
#ユーザ   システム       経過
#121.240      0.828   1551.608
save(driver_prob_rf, file="driver_prob_rf.RData")
load("driver_prob_rf.RData")
write.csv(driver_prob_rf, file="driver_prob_rf.csv", row.names=FALSE, quote=FALSE)


###############################
## RF by caret package with 3-fold CV 
###############################
library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(4, type = "SOCK"))
before <- proc.time()
#driver_prob_rf <- foreach(i=1:3, .combine="rbind") %dopar% {
driver_prob_rf <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  require(randomForest)
  require(caret)
  require(gbm)
  #require(e1071)
  calcProb2(id = driver_IDs[i], train_method="rf", N_trip=N_trip )
}
proc.time() - before
#ユーザ   システム       経過
#121.240      0.828   1551.608
save(driver_prob_rf, file="driver_prob_rf_with_cv.RData")
load("driver_prob_rf_with_cv.RData")
write.csv(driver_prob_rf, file="driver_prob_rf_with_cv.csv", row.names=FALSE, quote=FALSE)

###############################
## gbm by caret package with 3-fold CV 
###############################
library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(2, type = "SOCK"))
before <- proc.time()
driver_prob_gbm <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  require(randomForest)
  require(caret)
  require(gbm)
  calcProb2(id = driver_IDs[i], train_method="gbm", N_trip=N_trip )
}
proc.time() - before
#ユーザ   システム       経過
#121.240      0.828   1551.608
save(driver_prob_gbm, file="driver_prob_gbm_with_cv.RData")
load("driver_prob_gbm_with_cv.RData")
write.csv(driver_prob_gbm, file="driver_prob_gbm_with_cv.csv", row.names=FALSE, quote=FALSE)



calcProb2(id=1, train_method="rf")



j <- 20
tmp.id <- driver_IDs[j]
tmp.driver <- subset(dat.all1, driver_id == tmp.id)
tmp.true <- subset(driver_prob, driver_id == tmp.id & prob==1)
tmp.false <- subset(driver_prob, driver_id != tmp.id)
tmp.false <- tmp.false[sample(1:dim(tmp.false)[1], N_trip*3),]

tmp.true$target <- 1
tmp.false$target <- 0
tmp.dat <- rbind(tmp.true, tmp.false)

dat.model <- merge(tmp.dat[,c("driver_trip","target")], dat.all1, by="driver_trip", all.x=TRUE)
dim(dat.model)
fit <- glm(factor(target)~max_turn+avg_turn+median_turn+total_turn+
             max_speed+avg_speed+median_speed+max_acceleration+min_acceleration+
             avg_acceleration+median_acceleration+total_stop_sec, 
           data = dat.model, family=binomial(logit))
pred.driver <- predict(fit, newdata=tmp.driver)



library(randomForest)


j<-2000
before <- proc.time()
for(j in 1:N_driver){
    feature <- subset(dat.all1, driver_id==driver_IDs[j])
    feature.km <- kmeans(feature[,-c(1,2)],2)
    clust_count <- table(feature.km$cluster)
    majority_clust <- names(clust_count[rev(order(clust_count))])[1]
    tmp.prob <- ifelse(feature.km$cluster==majority_clust, 1, 0)
    tmp.trip <- paste(driver_IDs[j], 1:N_trip, sep="_")
    driver_prob[seq((j-1)*N_trip+1, j*N_trip), "driver_id"] <- driver_IDs[j]
    driver_prob[seq((j-1)*N_trip+1, j*N_trip), "trip_id"] <- seq(1, N_trip)
    driver_prob[seq((j-1)*N_trip+1, j*N_trip), "driver_trip"] <- tmp.trip
    driver_prob[seq((j-1)*N_trip+1, j*N_trip), "prob"] <- tmp.prob
}
proc.time - before

dim(driver_prob[seq((j-1)*N_trip+1, j*N_trip), ])
head(driver_prob[seq((j-1)*N_trip+1, j*N_trip), ])
table(driver_prob[seq((j-1)*N_trip+1, j*N_trip), "prob"])



with(driver_prob, table(driver_id, prob))
write.csv(driver_prob[,c("driver_trip","prob")], quote=FALSE,
          file="submission_kmeans.csv", row.names=FALSE)

#ERROR: Unable to find 166600 required key values in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_1' in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_2' in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_3' in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_4' in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_5' in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_6' in the 'driver_trip' column 
#ERROR: Unable to find the required key value '2737_7' in the 'driver_trip' column 
#ERROR: Unable to find 

#############################################################################
## 並列処理テスト
library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(2, type = "SOCK"))
before <- proc.time()
x <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  calcFeature(driver_id = files[i], feature_names=feature_names, N_trip=N_trip )
}
colnames(x) <- feature_names
proc.time() - before


before <- proc.time()
x <- foreach(i=1:20, .combine="rbind") %do% {
  calcFeature(driver_id = files[i], feature_names=feature_names, N_trip=N_trip )
}
colnames(x) <- feature_names
proc.time() - before


library(foreach)
library(doSNOW)
registerDoSNOW(makeCluster(4, type = "SOCK"))
before <- proc.time()
dat.comb <- foreach(i=1:N_driver, .combine="rbind") %dopar% {
  #print(paste(Sys.time(), " | Loop ",i, " is done...",sep=""))
  calcFeature(driver_id = files[i], feature_names=feature_names, N_trip=N_trip )
}
colnames(dat.comb) <- feature_names
proc.time() - before
save(dat.comb, file="dat.comb.RData")



