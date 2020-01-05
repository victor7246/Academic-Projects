library(conjoint)
library(AlgDesign)
data = read.csv("data/main.csv")
data_model = data[,-c(1,2,3)]
prof_data <- read.csv("data/prof.csv", header=TRUE)
level_data <- read.csv("data/levels.csv", header=TRUE)
pref_vector = data_model$Rating

model = lm(data_model$Rating~.-data_model$Rating, data_model)
summary(model)
test = data_model[,-14]
test = cbind(test,predict.lm(model,test),data_model[,14])
clustering = c()
n = nrow(data_model)
k = n/15
for (i in (1:k))
{
  n1 = 15*(i-1)+1
  n2 = 15*i
  dummy_data = data_model[n1:n2,]
  model_dummy = lm(dummy_data$Rating~.-dummy_data$Rating, dummy_data)
  clustering = rbind(clustering,model_dummy$coefficients)
}
hh = hclust(dist(clustering), method = "complete", members = NULL)
plot(hh)
clusters = kmeans(clustering,centers = 3)

cluster1 = c()
cluster2 = c()
cluster3 = c()



for (i in (1:k))
{
  if (clusters$cluster[i] == 1)
  {
    cluster1 = rbind(cluster1,clustering[i,])
  } 
  if (clusters$cluster[i] == 2)
  {
    cluster2 = rbind(cluster2,clustering[i,])
  } 
  if (clusters$cluster[i] == 3)
  {
    cluster3 = rbind(cluster3,clustering[i,])
  } 
}
write.csv(file = "data/cluster1.csv",cluster1)
write.csv(file = "data/cluster2.csv",cluster2)
write.csv(file = "data/cluster3.csv",cluster3)



Conjoint(pref_vector, prof_data, level_data)

#imp<-caImportance(pref_vector, prof_data)
