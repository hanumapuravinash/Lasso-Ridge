#LOAD DATA
titanic.train <- read.csv("C:/Users/DiMaria/Desktop/ATI/Kaggle/Lesso Ridge Titanic/train.csv", stringsAsFactor=FALSE)
titanic.test <- read.csv("C:/Users/DiMaria/Desktop/ATI/Kaggle/Lesso Ridge Titanic/test.csv", stringsAsFactor=FALSE)

#LOAD R LIBRARIES
library(plyr)
library(rpart)
library(caret)
library(caTools)
library(mice)
library(stringr)
library(Hmisc)
library(ggplot2)
library(vcd)
library(ROCR)
library(pROC)
library(VIM)
library(glmnet)

## EDA ##
str(titanic.train)
summary(titanic.train)
table(titanic.train$Survived, titanic.train$Sex)

#Did age influence survival?
ggplot(titanic.train, aes(x=Age, y=PassengerId, color = as.factor(Survived))) +                      
  geom_point() + 
  facet_grid(Sex ~.) +
  ggtitle("Survival vs Passenger's Age")+
  xlab("Age") + 
  theme(legend.position = "none")+
  scale_colour_manual(values = c("#FF0000","#0000FF"))

#Is survival of a passenger related to his/her Pclass and port of embarkation?
ggplot(titanic.train[titanic.train$Embarked != "",], aes(x=Embarked, y=PassengerId)) +  
  geom_tile(aes(fill = as.factor(Survived))) + 
  facet_grid(. ~ Pclass) +
  ggtitle("Survival vs Passenger's Pclass and Port of Embarkation")+
  theme(legend.position = "none")+
  scale_fill_manual(values = c("#FF0000","#0000FF"))

ggplot(titanic.train[titanic.train$Embarked != "",], aes(x=Embarked, y=PassengerId)) +  
  geom_tile(aes(fill = as.factor(Survived))) + 
  facet_grid(. ~ Sex) +
  ggtitle("Survival vs Passenger's Sex and Port of Embarkation")+
  theme(legend.position = "none")+
  scale_fill_manual(values = c("#FF0000","#0000FF"))

# Did travelling with the family mattered?
mosaic(~ Sex + (Age > 15) + (SibSp + Parch > 0) + Survived, data = titanic.train[complete.cases(titanic.train),],
       shade=T, legend=T)

Survived = titanic.train$Survived
titanic.test$Survived = NA
all = rbind(titanic.train, titanic.test)

## FEATURE ENGINEERING ##

# Title

 #// Names of the passengers consist of titles ascribed to each individual as 
 # per their gender, age and social status, which can be extracted and maximum and minimum 
 # age for each category can be enumerated //#

all$Title = sapply(all$Name,function(x) strsplit(x,', ')[[1]][2])
all$Title = sapply(all$Title,function(x) strsplit(x,'\\. ')[[1]][1])

as.data.frame(
  cbind("Title" = unique(all$Title), 
        "No_of_passengers" = sapply(unique(all$Title), function(x) nrow(all[all$Title == x,])),
        "Age_missing" = sapply(unique(all$Title), function(x) nrow(all[all$Title == x & is.na(all$Age),])),
        "Minimum_Age" = sapply(unique(all$Title), function(x) min(all[all$Title == x,'Age'], na.rm = TRUE)),
        "Maximum_Age" = sapply(unique(all$Title), function(x) max(all[all$Title == x,'Age'], na.rm = TRUE))), row.names = F)

 # 18 categories can be combined into more manageable 5 categories as per their gender and age:

 #   Mr:     For men above 14.5 years
 #   Master: For boys below and equal to 14.5 years
 #   Miss:   For girls below and equal to 14.5 years
 #   Ms:     For women above 14.5 years, maybe unmarried
 #   Mrs:    For married women above 14.5 years

all[(all$Title == "Mr" & all$Age <= 14.5 & !is.na(all$Age)),]$Title = "Master"

all[all$Title == "Capt"|
      all$Title == "Col"|
      all$Title == "Don"|
      all$Title == "Major"|
      all$Title == "Rev"|      
      all$Title == "Jonkheer"|
      all$Title == "Sir",]$Title = "Mr"

 # None of these women are travelling with family, hence can be categorised as single women for this analysis
all[all$Title == "Dona"|
      all$Title == "Mlle"|
      all$Title == "Mme",]$Title = "Ms"

 # Categories Lady and Countess as a married woman
all[all$Title == "Lady"| all$Title == "the Countess",]$Title = "Mrs"

 # Categorise doctors as per their sex
all[all$Title == "Dr" & all$Sex == "female",]$Title = "Ms"
all[all$Title == "Dr" & all$Sex == "male",]$Title = "Mr"

all$Title = as.factor(all$Title)
all$Title <- droplevels(all$Title)
summary(all$Title)

 ## Master   Miss     Mr    Mrs     Ms 
 ##     66    260    777    199      7

# Family Size
all$FamilySize = ifelse(all$SibSp + all$Parch + 1 <= 3, 1,0) # Small = 1, Big = 0

# Mother
all$Mother = ifelse(all$Title=="Mrs" & all$Parch > 0, 1,0)

# Single
all$Single = ifelse(all$SibSp + all$Parch + 1 == 1, 1,0) # People travelling alone

# FamilyName
all$FamilyName = sapply(all$Name,function(x) strsplit(x,', ')[[1]][1])

Family.Ticket = all[all$Single == 0,c("FamilyName", "Ticket")]
Family.Ticket = Family.Ticket[order(Family.Ticket$FamilyName),]
head(Family.Ticket)

all$FamilyName  = paste(all$FamilyName , str_sub(all$Ticket,-3,-1), sep="")

# Family Survived
all$FamilySurvived = 0
 # Dataset of passengers with family
Families = all[(all$Parch+all$SibSp) > 0,]

 # Group families by their family name and number of survivals in the family
Survival.GroupByFamilyName = aggregate(as.numeric(Families$Survived), by=list("FamilyName" = Families$FamilyName), FUN=sum, na.rm=TRUE)

 # Family is considered to have survived if atleast one member survived
FamilyWithSurvival = Survival.GroupByFamilyName[Survival.GroupByFamilyName$x > 0,]$FamilyName
all[apply(all, 1, function(x){ifelse(x["FamilyName"] %in% FamilyWithSurvival,TRUE,FALSE)}),]$FamilySurvived = 1

#Age Class
all$AgeClass = ifelse(all$Age<=10,1,
                      ifelse(all$Age>10 & all$Age<=20,2,
                             ifelse(all$Age>20 & all$Age<=35,3,4)))
all$AgeClass = as.factor(all$AgeClass)