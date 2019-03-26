library(haven)
library(caret)
library(psych)
library(ggthemes)
library(rpart)
library(dplyr)
#library(forcats)
setwd('desktop')
train <- read.csv('train.csv',stringsAsFactors = F)
test <- read.csv('test.csv',stringsAsFactors = F)
View(train)
describe(train)
describe(test)
full <- bind_rows(train,test)
describe(full)
str(full)

#3.Visualization
#sibsp and survival plot
ggplot(full[1:891,], aes(x = SibSp,fill=factor(Survived))) +
  geom_bar(position='fill') +
  # scale_x_continuous(breaks=c(1:11)) +
  #scale_y_continuous(limits=c(0,150)) +
  labs(x = 'Sibsp',y='Probability')+
  scale_fill_manual(values=c("#737373","#262626"))+
  guides(fill=guide_legend(title="Survived"))
#age and survival 
ggplot(full[1:891,], aes(x = Age,fill=factor(Survived))) +
  geom_histogram(aes(y=..density..))+
  # scale_x_continuous(breaks=c(1:11)) +
  #scale_y_continuous(limits=c(0,150)) +
  labs(x = 'Age',y='')+
  scale_fill_manual(values=c("#262626","#737373"))+
  facet_grid(.~Survived,labeller = label_both)+
  guides(fill=guide_legend(title="Survived"))

full[1:891,] %>% 
  ggplot(aes(x=Age, y=..density..)) +
  geom_density(alpha=0.3,aes(fill=as.factor(Survived)))+
  scale_x_continuous(limits=c(0,80))+
  guides(fill=guide_legend(title="Survived"))
#fare with log transformation
p <- ggplot(full[1:891,], aes(x = Fare)) +
  geom_histogram(aes(y=..density..))+
  # scale_x_continuous(breaks=c(1:11)) +
  #scale_y_continuous(limits=c(0,150)) +
  scale_x_log10()+
  labs(x = 'Fare',y='')
p+
  geom_density(data=full[1:891,],aes(x=Fare,y=..density..),fill='blue',alpha=0.1)+
  # scale_x_continuous(breaks=c(1:11)) +
  #scale_y_continuous(limits=c(0,150)) +
  scale_x_log10()+
  labs(x = 'Fare',y='')

#sex and class
ggplot(full[1:891,], aes(x = Pclass,fill=factor(Survived))) +
  geom_bar(position='fill') +
  # scale_x_continuous(breaks=c(1:11)) +
  #scale_y_continuous(limits=c(0,150)) +
  labs(x = 'Sex',y='Probability')+
  scale_fill_manual(values=c("#737373","#262626"))+
  facet_grid(.~Sex)+
  guides(fill=guide_legend(title="Survived"))

 #familysize plot
full$fsize <- full$SibSp + full$Parch + 1
ggplot(full[1:891,], aes(x = fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  scale_y_continuous(limits=c(0,100)) +
  labs(x = 'Family Size')+
  scale_fill_manual(values=c("#262626","#737373"))+
  facet_grid(.~Sex)+
  guides(fill=guide_legend(title="Survived"))


#Missing Value
#deal with NA Embarked, Fare, Age
full[full$Embarked=='','Embarked'] <- 'S'
meanfare <- mean(train$Fare,na.rm=T)
full[is.na(full$Fare),'Fare'] <- meanfare

age=rpart(Age ~Pclass+Sex+SibSp+Parch+Fare+Embarked,data=full[!(is.na(full$Age)),],method="anova")
full$Age[is.na(full$Age)]=predict(age,full[is.na(full$Age),])

#add new feature family size
full$FsizeD[full$fsize == 1] <- 'single'
full$FsizeD[full$fsize < 4 & full$fsize > 1] <- 'small'
full$FsizeD[full$fsize < 7 & full$fsize > 3] <- 'medium'
full$FsizeD[full$fsize > 6] <- 'large'
t1 <- full[1:891,]
# Show family size by survival using a mosaic plot
tabpct(t1$FsizeD, t1$Survived,col=c("#262626","#737373"),main='Family Size by Survival',xlab='family size',ylab='survived')
mosaicplot(table(t1$FsizeD, t1$Survived), main='Family Size by Survival', col=T)

#add new feature title
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title)
full <- full%>% 
  mutate(Title = factor(Title)) %>%
  mutate(Title = fct_collapse(Title, "Miss" = c("Mlle", "Ms"), "Mrs" = "Mme", 
                              "Ranked" = c( "Major", "Dr", "Capt", "Col", "Rev"),
                              "Royalty" = c("Lady", "Dona", "the Countess", "Don", "Sir", "Jonkheer")))
levels(full$Title)
#add new feature Agetype
full$Aget[full$Age <= 5] <- 'Child'
full$Aget[full$Age > 5 & full$Age<= 18] <- 'Teenager'
full$Aget[full$Age > 18 & full$Age<= 25] <- 'Adult'
full$Aget[full$Age > 25 & full$Age<= 40] <- 'Elder'
full$Aget[full$Age > 40 ] <- 'Old'
full$Aget  <- factor(full$Aget)
#add new feature mother
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age >= 18 & full$Title != 'Miss'] <- 'Mother'
full$Mother <- factor(full$Mother)


full <- full %>% 
  mutate(Pclass=as.factor(Pclass),Sex=as.factor(Sex),Embarked=as.factor(Embarked),Survived=as.factor(Survived),FsizeD=as.factor(FsizeD))
full <- full %>% 
  select(2,3,5,6,7,8,10,12,13,14,15,16,17)


train1 <- full[1:891,]
train1$Title <- as.factor(train1$Title)
test1 <- full[892:1309,]
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
# a) linear algorithms

set.seed(7)
fit.lda <- train(Survived~., data=train1, method="lda", metric=metric, trControl=control)
set.seed(7)
fit.cart <- train(Survived~., data=train1, method="rpart", metric=metric, trControl=control)
set.seed(7)
fit.knn<- train(Survived~., data=train1, method="knn", metric=metric, trControl=control)
set.seed(7)
fit.svm<- train(Survived~., data=train1, method="svmRadial", metric=metric, trControl=control)
set.seed(7)
fit.brnn<- train(Survived~., data=train1, method='nnet', metric=metric, trControl=control)
set.seed(7)
fit.nb<- train(Survived~., data=train1, method='nb', metric=metric, trControl=control)
set.seed(7)
fit.rf<- train(Survived~., data=train1, method="rf", metric=metric, trControl=control)
results <- resamples(list(lda=fit.lda,cart=fit.cart, knn=fit.knn,svm=fit.svm,rf=fit.rf,rf=fit.rf,nnet=fit.brnn))
summary(results)
dotplot(results)
print(fit.lda)
set.seed(7)
predictions <- predict(fit.brnn, newdata=test1)
write.csv(predictions,'p.csv')
#confusionMatrix(predictions, test1$Survived)

predict(fit.lda, newdata=test1)
summary(results)


set.seed(7)
rf <- randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+fsize+FsizeD+Title+Aget+Mother,data=train1)
plot(rf)
legend('topright', colnames(rf$err.rate),  fill=1:3)
importance    <- importance(rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
 
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
