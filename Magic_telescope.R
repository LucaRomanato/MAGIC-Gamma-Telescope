install.packages("e1071")
install.packages("ROCR")
install.packages("MLmetrics")
install.packages("DataExplorer")
install.packages("keras")
install.packages("dplyr")
install.packages("plyr")
install.packages("caret")
install.packages("pROC")
install.packages("FactoMineR")
install.packages("factoextra")
library(FactoMineR)
library(factoextra)
library(DataExplorer)
library(MLmetrics)
library(ROCR)
library(e1071)
library(devtools)
library(keras)
library(plyr)
library(dplyr)
library(caret)
library(pROC)
#import dataset e ordine sulla fLength per mischiare l'ordine della classe target
df=read.csv(file="Dataset.csv", sep = ",")
df = df[order(df$fLength),]
head(df)
df2 = df
#PCA e analisi esplorativa
names(df[,-1])
plot_intro(df[,-1])
plot_bar(df[,-1])
plot_histogram(df[,-1])
#create_report(df[,2:11])
res.pca <- PCA(df[,2:11], graph = FALSE) 
eig.val <- get_eigenvalue(res.pca)
eig.val
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50)) 
var <- get_pca_var(res.pca)
fviz_pca_var(res.pca, col.var = "black") 

#riduzione dataset per elementi che ci interessano (zone 3d, indice, invece che ratio highest tengo ratio sum 2 highest)
#trasformo g in 0 e h in 1 in modo da non aver problemi di conversione con le stringhe\char
df <- data.frame(df, stringsAsFactors = FALSE)
lookup <- c("g" = 0, "h" = 1)
df$class_n <- lookup[df$class]
df = df[ , -which(names(df) %in% c("fM3Long","fM3Trans", "X", "fConc1", "class"))]
#analisi esplorativa dei campi
head(df)
#create_report(df)

#---------------------------------------------------------------------
#rete neurale

# Build your own `normalize()` function
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

# Normalize tutte le colonne tranne la target che la metto come numerica
df_norm <- as.data.frame(lapply(df[1:7], normalize))

df[,8] <- as.numeric(df[,8])

# Turn  data into a matrix altrimenti keras non va
df <- as.matrix(df)

# Set `dimnames` to `NULL`
dimnames(df) <- NULL

# Determine sample size per dividere in train, test, val
ind <- sample(2, nrow(df), replace=TRUE, prob=c(0.80, 0.20))

# Split the data
df.training <- df[ind==1,]
df.test <- df[ind==2, 1:7]
# Split the class attribute
df.testtarget <- df[ind==2, 8]
# One hot encode target values, categorical avendo 2 neuroni in output, conseguente softmax e categorical_crossentropy
df.testLabels <- to_categorical(df.testtarget)

#kfold cv 
set.seed(450)
auc.cv <- NULL
accu.cv = NULL
prec.cv = NULL
rec.cv = NULL
f1.cv = NULL
vv.cv = NULL
fv.cv = NULL
nv.cv = NULL
ff.cv = NULL

k <- 10

pbar <- create_progress_bar('text')
pbar$init(k)

#Randomly shuffle 
df.training<-df.training[sample(nrow(df.training)),]
#Create 10 equally size folds
folds <- cut(seq(1,nrow(df.training)),breaks=10,labels=FALSE)

#10 fold cross validation
for(i in 1:k){
  #Segementdata by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  train.cv <- df.training[-testIndexes,1:7]
  trainL.cv <- df.training[-testIndexes,8]
  val.cv <- df.training[testIndexes,1:7]
  valL.cv <- df.training[testIndexes,8]
  df.trainLabels <- to_categorical(trainL.cv)
  df.valLabels <- to_categorical(valL.cv)

  # Initialize a sequential model
  model <- keras_model_sequential() 
  
  # Add layers to the model
  model %>% 
    layer_dense(units = 64, activation = 'relu',  input_shape = dim(train.cv)[2]) %>% 
    layer_dense(units = 20, activation = 'relu') %>%
    layer_dense(units = 2, activation = 'softmax')
  
  # Print a summary of a model
  summary(model)
  
  # Compile the model
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )
  
  # Fit the model 
  model %>% fit(
    train.cv, 
    df.trainLabels, 
    epochs = 50, 
    batch_size = 128, 
    validation_data = list(val.cv, df.valLabels)
  )
  
  #auc e roc
  pred_nn = model %>% predict(df.test)
  pred_nn = pred_nn[,1]
  auc.cv[i] = roc(df.testLabels[,2],pred_nn)$auc
  
  #precision recall f1 accuracy
  classes = model %>% predict_classes(df.test, batch_size = 128)
  metrics = confusionMatrix(as.factor(classes), as.factor(df.testtarget))
  accu.cv[i] = metrics$overall[1]
  prec.cv[i] = metrics$byClass[5]
  rec.cv[i] = metrics$byClass[6]
  f1.cv[i] = metrics$byClass[7]
  
  #matrice confusione
  vv.cv[i] = metrics$table[1]
  fv.cv[i] = metrics$table[2]
  nv.cv[i] = metrics$table[3]
  ff.cv[i] = metrics$table[4]
  
  
  pbar$step()
}

auc = mean(auc.cv)
plot(roc(df.testLabels[,2],pred_nn), print.auc = auc)
accu = mean(accu.cv)
prec = mean(prec.cv)
rec = mean(rec.cv)
f1 = mean(f1.cv)
accu
auc
prec
rec
f1

#matrice
vv = as.integer(mean(vv.cv))
fv = as.integer(mean(fv.cv))
nv = as.integer(mean(nv.cv))
ff = as.integer(mean(ff.cv))
CM = matrix( c(vv,fv,nv,ff),  nrow=2, ncol=2)
CM

#---------------------------------------------------------------------
#random forest

#riduzione dataset per elementi che ci interessano (zone 3d, indice, invece che ratio highest tengo ratio sum 2 highest)
df2 <- data.frame(df2, stringsAsFactors = FALSE)
df2 = df2[ , -which(names(df2) %in% c("fM3Long","fM3Trans", "X", "fConc1"))]

ind2 <- sample(2, nrow(df2), replace=TRUE, prob=c(0.80, 0.20))

# Split the data
df2.training <- df2[ind2==1,]
df2.test <- df2[ind2==2,]

#create objects x which holds the predictor variables and y which holds the response variables
x = df2.training[,-8]
y = df2.training$class

trc = trainControl(method='cv',number=10, summaryFunction=multiClassSummary, classProbs=T, savePredictions = T)
rfFit = train(x,y, method = "rf", ntree = 64, trControl=trc)
rfFit
#Predict testing set
Predict <- predict(rfFit,df2.test)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(Predict, df2.test$class)

#ROC

selectedIndices <- rfFit$pred$mtry == 2
# Plot:
roc_rf = roc(rfFit$pred$obs[selectedIndices], rfFit$pred$g[selectedIndices])
plot(roc_rf, print.auc=TRUE)
