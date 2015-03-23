# setwd("/Users/okada/myWork/kaggle/driver/")
dat <- read.csv("./data/drivers/1/1.csv", header=TRUE)



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

feature <- c(max.turn, avg.turn, med.turn, tot.turn, max.speed, avg.speed, med.speed, 
             max.accl, min.accl, avg.accl, med.accl)

####################################################################

# N_driver <- 3612
files <- list.files("./data/drivers/")
for(i in 1:length(files)){
  datafiles <- list.files(paste("./data/drivers/", files[i], "/", sep=""), pattern="csv")
}

N_driver <- length(files)
N_trip <- 200
N_feature <- 11
dat.all <- as.data.frame(matrix(rep(NA_real_, (N_feature+2)*N_trip*N_driver), ncol=(N_feature+2)))
feature_names <- c("driver_id","trip_id", "max_turn", "avg_turn", "median_turn", 
                   "total_turn","max_speed","avg_speed","median_speed",
                   "max_acceleration","min_acceleration","avg_acceleration", 
                   "median_acceleration") 
colnames(dat.all) <- feature_names

for(j in 1:N_driver){
  datafiles <- list.files(paste("./data/drivers/", files[j], "/", sep=""), pattern="csv")
  #N_trip <- length(datafiles)
  feature <- as.data.frame(matrix(rep(NA_real_, 
                                      length(feature_names)*N_trip*1), 
                                  ncol=length(feature_names)))
  colnames(feature) <- feature_names
  for(i in 1:N_trip){
    dat <- read.csv(paste("./data/drivers/",files[j], "/", datafiles[i], sep=""), header=TRUE)
    
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
    
    dat.all[(j-1)*N_trip+i, ] <- c(j, i, max.turn, avg.turn, med.turn, tot.turn, 
                                   max.speed, avg.speed, med.speed, max.accl, 
                                   min.accl, avg.accl, med.accl)
  }
  print(paste(j,"th file is done...",sep=""))
}
save(dat.all, "dat.all.RData")
load("dat.all.RData")

###############
## Model Fit ##
###############

driver_prob <- data.frame(driver_id=rep(NA_real_, N_trip*N_driver),
                          trip_id = rep(NA_real_, N_trip*N_driver),
                          driver_trip=rep(NA_real_, N_trip*N_driver),
                          prob=rep(NA_real_, N_trip*N_driver))
for(j in 1:N_driver){
  feature <- subset(dat.all, driver_id==j)
  feature.km <- kmeans(feature[,-c(1,2)],2)
  clust_count <- table(feature.km$cluster)
  majority_clust <- names(clust_count[rev(order(clust_count))])[1]
  tmp.prob <- ifelse(feature.km$cluster==majority_clust, 1, 0)
  tmp.trip <- paste(j, 1:N_trip, sep="_")
  driver_prob[seq((j-1)*N_trip+1, j*N_trip), "driver_id"] <- j
  driver_prob[seq((j-1)*N_trip+1, j*N_trip), "trip_id"] <- seq(1, N_trip)
  driver_prob[seq((j-1)*N_trip+1, j*N_trip), "driver_trip"] <- tmp.trip
  driver_prob[seq((j-1)*N_trip+1, j*N_trip), "prob"] <- tmp.prob
}
head(driver_prob)
with(driver_prob, table(driver_id, prob))
write.csv(driver_prob[,c("driver_trip","prob")], quote=FALSE,
          file="submission_kmeans.csv", row.names=FALSE)



summary(feature)
pairs(feature[,-c(1,2)])
feature.km <- kmeans(feature[,-c(1,2)],2)
clust_count <- table(feature.km$cluster)
# cluster with the largest count will get the probability 1, 0 otherwise.
majority_clust <- names(clust_count[rev(order(clust_count))])[1]
driver_prob <- ifelse(feature.km$cluster==majority_clust, 1, 0)
driver_trip <- paste("1",1:200,sep="_")


feature$cluster <- feature.km$cluster
head(feature)
library(plyr)
ddplyr(feature, .(cluster), summarize, )


v1[1,] %*% v2[1,]
v1[1,] %*% v1[1,]
18.6^2+11.1^2
v1[1,] %*% v2[1,]/((v1[1,]%*%v1[1,])^0.5*(v2[1,]%*%v2[1,])^0.5)
acos(v1[1,] %*% v2[1,]/((v1[1,]%*%v1[1,])^0.5*(v2[1,]%*%v2[1,])^0.5))

(sqrt(diag(v1%*%t(v1)))*sqrt(diag(v2%*%t(v2))))[1:5]
diag(v1%*%t(v2))[1:5]


acos(0)

dat1 <- dat[-c(tot.sec[1]-1, tot.sec[1]),]
dat2 <- dat[-c(1, tot.sec[1]),]
dat3 <- dat[-c(1,2),]
vec1 <- dat2-dat1
vec2 <- dat3-dat2

calcMaxSpeed <- function(dat){
}
calcTotalDistance <- function(dat){
}
calcAvgSpeed <- function(dat){
}
calcMaxTurn <- function(dat){
}
calcTotalTurn <- function(dat){
}
calcTotalStopSeconds <- function(dat){
}




