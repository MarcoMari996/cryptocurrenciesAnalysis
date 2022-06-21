rm(list = ls())

## importing libraries ####
library(ggthemes)
library(ggplot2)
library(data.table)
library(readxl)
library(ppcor)
library(corrplot)
library(DescTools)
library(dplyr)
library(Hmisc)
library(pastecs)
library(qgraph)
library(igraph)
library(nnet)
library(neuralnet)
library(e1071)
library(cluster)
library(moments)


## importing data ####
data <- read.csv('./dataset.csv', sep = ',')
data$Date <- as.Date(data$Date)
data$XLM <-sapply(data$XLM, function(x) ifelse(x>10*mean(data$XLM), x/100, x))
data$XRP <-sapply(data$XRP, function(x) ifelse(x>10*mean(data$XLM), x/100, x))
data$USD.EUR <- sapply(data$USD.EUR, function(x) ifelse(x>100, x/1000, x))
data$OIL <- sapply(data$OIL, function(x) ifelse(x<0, x <- NA, x))

## handling na values ####
df <- as.data.frame(sapply(data[-1], function(x) nafill(x, type = 'locf')))
# the last date has no more values available forward on, hence the na value is 
# replaced with the last non-na available value
df <- as.data.frame(sapply(df, function(x) nafill(x, type = 'nocb')))

crypto <- as.data.frame(cbind(data$Date, df))
colnames(crypto)[1] <- 'Date'

crypto$XLM <- NULL
df$XLM <- NULL
## creating the retunrs vector #### 
df <- as.data.frame(sapply(df, function(x) diff(log(x), lag=1)))
df <- as.data.frame(sapply(df, function(x) nafill(x, type = 'locf')))
df <- as.data.frame(sapply(df, function(x) nafill(x, type = 'nocb')))
returns <- as.data.frame(cbind(data$Date[-1], df))
colnames(returns)[1] <- 'Date'

## descriptive statistics ####
stats <- as.data.frame(sapply(crypto[-1], function(x) cbind(range(x)[2]-range(x)[1], max(x),var(x),sd(x), mean(x))))
rownames(stats) <- c('Range', 'Max', 'Var', 'Std', 'Mean')
print(stats)

## plotting prices ####
ggplot(crypto, aes(Date))+
  geom_line(aes(y=BTC, colour="BTC"), size = 1.2)+
  geom_line(aes(y=ETH, colour="ETH"), size = 1.2)+
  geom_line(aes(y=XRP, colour="XRP"))+
  geom_line(aes(y=LTC, colour="LTC"))+
  # geom_line(aes(y=XLM, colour="XLM"))+ 
  labs(y="", x='Date', 
       color='Cryptocurrency: ',
       title = "Cryptocurrencies Prices", 
       subtitle = "from 1-1-2016 to 29-12-2020")+
  theme_fivethirtyeight()+
  theme(axis.title = element_text(), 
        legend.text = element_text(size = 12))

## plotting returns ####
ggplot(returns, aes(Date))+
  geom_line(aes(y=BTC, colour="BTC"), alpha=0.5)+
  geom_line(aes(y=ETH, colour="ETH"), alpha=0.5)+
  # geom_line(aes(y=XRP, colour="XRP"), alpha=0.5)+
  geom_line(aes(y=LTC, colour="LTC"), alpha=0.5)+
  geom_line(aes(y=XLM, colour="XLM"), alpha=0.5)+ 
  labs(y="Return", x='Date',
       title = "Cryptocurrencies Returns", 
       color="Cryptocurrency: ",
       subtitle = "from 1-1-2016 to 29-12-2020")+
  theme_fivethirtyeight()+
  theme(axis.title = element_text())+
  scale_color_brewer(palette = 'Dark2')

## plotting financial indices ####
ggplot(crypto, aes(Date))+
  geom_line(aes(y=NASDAQ, colour="Nasdaq"), size = 1.3)+
  geom_line(aes(y=DJI, colour="DowJones"), size = 1.3)+
  geom_line(aes(y=N225, colour="Nikkei"), size = 1.3)+
  geom_line(aes(y=HSI, colour="Hang Seng"), size = 1.3)+
  geom_line(aes(y=SSEC, colour="Shangai"), size = 1.3)+
  geom_line(aes(y=STOXX50E, colour="Stoxx50e"), size = 1.3)+
  geom_line(aes(y=BTC, colour='BTC'), alpha=.7)+  
  geom_line(aes(y=GOLD, colour='Gold'), size = 1.3)+
  labs(y="", x='Date',
       title = "Main Stock Market Indices", 
       color="Index: ",
       subtitle = "from 1-1-2016 to 29-12-2020")+
  theme_fivethirtyeight()+
  theme(axis.title = element_text(), 
        legend.text = element_text(size = 12))

## plotting Tech companies indices ####
ggplot(crypto, aes(Date))+
  geom_line(aes(y=BTC/50, color="BTC (scaled down)"), alpha=.6, size = 1.2)+
  geom_line(aes(y=NVDA, colour="Nvidia"), size = 1.1)+
  geom_line(aes(y=INTC, colour="Intel"), size = 1.2)+
  geom_line(aes(y=AMD, colour="AMD"), size = 1.2)+
  geom_line(aes(y=PYPL, colour="PayPal"), size = 1.2)+
  labs(y="", x='Date',
       title = "Tech Companies (principal GPU manifacturers) Indices", 
       color="Index: ",
       subtitle = "from 1-1-2016 to 29-12-2020")+
  theme_fivethirtyeight()+
  theme(axis.title = element_text(), 
        legend.text = element_text(size = 12))

## plotting main FIATs ####
ggplot(crypto, aes(Date))+
  geom_line(aes(y=USD.EUR, color='USD-EUR'))+
  geom_line(aes(y=USD.CNY, color='USD-CNY'))+
  geom_line(aes(y=USD.JPY, color='USD-JPY'))+
  labs(y='', x='Date', title = 'Main FIATs')+
  theme_fivethirtyeight()+ theme(axis.title = element_text())

## plotting main ETFs ####
ggplot(crypto, aes(Date)) +
  geom_line(aes(y=ACWI.ETF, color='All Country World Index (ACWI)'), size = 1.2) +
  geom_line(aes(y=ICLN.ETF, color='IShares global CLeaN energy (ICLN)'), size = 1.2) +
  geom_line(aes(y=QQQ.ETF, color='Invesco PowerShares QQQ'), size = 1.2) +
  geom_line(aes(y=SMH.ETF, color='VanEck Vectors Semiconductor (SMH)'), size = 1.2)+
  geom_line(aes(y=XLF.ETF, color='Financial Select Sector SPDR Fund (XLF)'), size = 1.2)+
  geom_line(aes(y=BTC/70, color='BTC'), alpha=.5, size =1.1) +
  labs(y='', x='Date',
       color = '',
       title = 'Main Exchanged Traded Founds')+
  theme_fivethirtyeight() + 
  theme(axis.title = element_text(),
        legend.direction = "vertical",
        legend.position = "bottom",
        legend.text = element_text(size = 10))


## correlations ####
correlations <- cor(crypto[-1])
corrplot(correlations,
         # title = 'Correlation Matrix',
         method="circle",bg="black",
         addgrid.col = 'white',win.asp = .78,
         type='lower', diag=FALSE)

correlations <- cor(returns[-1])
corrplot(correlations,
         # title = 'Correlation Matrix w/Returns',
         bg='black', addgrid.col = 'white',
         method="circle", win.asp = .78,
         type='lower', diag=FALSE)

## Linear Models ####
full_lm <- lm(NASDAQ ~ ., data=returns[-1] )
summary(full_lm)
step_model <- step(full_lm, direction = 'both')
summary(step_model)
null_lm <- lm(NASDAQ ~ 1, data = returns[-1])
summary(null_lm)
anova(null_lm, step_model, full_lm)
## unusefull testing Lin Reg ####
fitted <- full_lm$fitted.values
results <- cbind.data.frame(returns$Date, returns$BTC, fitted)
colnames(results) <- c('Date', 'Observed', 'Fitted')
rmse <- sqrt(mean((results$Observed - results$Fitted)^2))

ggplot(results, aes(Date)) + 
  geom_line(aes(y = Observed, colour = "Observed", alpha=.3)) + 
  geom_line(aes(y = Fitted, colour = "Fitted", alpha=.3)) +
  labs(y='Returns', x='Date')+
  theme_fivethirtyeight() + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title = element_text())


train <- crypto[crypto$Date < as.Date("2019-08-01"),]
test <- crypto[crypto$Date >= as.Date("2019-08-01"),]

lin_reg <- lm(GOLD ~ ., data = train[-1])
test_pred <- predict(lin_reg, test)
results           <- cbind.data.frame(test$Date,test$GOLD,test_pred) 
colnames(results) <- c('Date','Real','Predicted')
results           <- as.data.frame(results)

ggplot(train, aes(Date)) + 
  geom_line(aes(y = GOLD, colour = "Observed", alpha=.3)) + 
  geom_line(aes(y = lin_reg$fitted.values, colour = "Fitted", alpha=.3)) +
  labs(y='Prices', x='Date',
       title = 'Train') +
  theme_fivethirtyeight() + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title = element_text())

ggplot(results, aes(Date)) + 
  geom_line(aes(y = Real, colour = "Observed", alpha=.3)) + 
  geom_line(aes(y = Predicted, colour = "Fitted", alpha=.3)) +
  labs(y='Prices', x='Date',title = 'Test')+
  theme_fivethirtyeight() + 
  theme(plot.title = element_text(hjust = 0.5), 
        axis.title = element_text())


SSE = sum((results$Real - results$Predicted)^2)
RMSE = sqrt(mean((results$Real - results$Predicted)^2))

## Network Analysis ####

# Simple correlation network
cor_network <- cor_auto(returns[-1])
Graph_1 <- qgraph(cor_network, graph = "cor", 
                  theme = "Borkulo", bg="lightgray",
                  layout = "spring", edge.width=1.5)
summary(Graph_1)

# Partial correlation network
Graph_2 <- qgraph(cor_network, graph= "pcor", 
                  theme = "Borkulo", bg="lightgray",
                  layout = "spring", edge.width=1.5)
summary(Graph_2)

# We can get more precise results if we eliminate links that are not statistically significant.
# The threshold argument can be used to do just that -- to remove edges that are not significant.
Graph_3 <- qgraph(cor_network, graph = "pcor",
                  layout = "spring", edge.width=1.5,
                  theme = "Borkulo", bg = 'lightgray',
                  threshold = "sig",sampleSize = nrow(data), alpha = 0.05)
summary(Graph_3)

# Investigate the centrality measures of the graphs 
centralities_Graph1 <- centrality(Graph_1, all.shortest.paths = TRUE)
centralities_Graph2 <- centrality(Graph_2)
centralities_Graph3 <- centrality(Graph_3)

# Plotting the centrality measures
centralityPlot(Graph_3)

# Compare the two networks
centralityPlot(GGM = list(correlation = Graph_1, partial_correlation = Graph_3),
              theme_bw = TRUE, include = c("Closeness"), 
              decreasing = T)

## Clustering (K-Means)  & Hierarchical) ####

set.seed(999)
clusters <- 20
wgssplot <- function(data, nclus, seed) {
  wgss <- rep(0, nclus)
  wgss[1] <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nclus) {
    set.seed(seed)
    wgss[i] <- sum(kmeans(data, centers=i)$withinss)
  }
  wgss <- as.data.frame(cbind(wgss, 1:nclus))
  colnames(wgss) <- c("wgss", "clusters")
  wgss$clusters <- as.integer(wgss$clusters)
  ggplot(data = wgss, aes(clusters, y = wgss),) +
    geom_line(linetype= "dashed") +
    geom_point(size = 3, color = "black", fill = 'white') +
    labs(x='Clusters', y='Within Clusters Sum of Squares', 
         title = 'Comparing Within Clusters\n Sum of Squares') + 
    theme_fivethirtyeight() + 
    theme(axis.title = element_text())
}
wgssplot(returns[-1], nclus = clusters, seed = 999)

kfit <- kmeans(scale(returns[-1]), 3, iter.max = 20)
df <- cbind.data.frame(scale(returns[-1]), kfit$cluster) 
colnames(df)[length(df)] <- "cluster"
clus <- list()

for (c in 1:max(df$cluster)) {
  clus[[c]] <- as.data.frame(
    df[df[colnames(df) == "cluster"] == c, 1:(length(colnames(df))-1)]
  )
  clus[[c]] <- as.data.frame(colMeans(clus[[c]]))
  colnames(clus[[c]]) <- "mean"
}
means <- data.frame(clus[1:length(clus)])
means <- data.frame(t(means))
feature <- NULL
clusters <- NULL
values <- NULL
for (f in colnames(means)) {
  feature <- c(feature, rep(f, length(clus)))
  clusters <- c(clusters, 1:length(clus))
  values <- c(values, means[, colnames(means) == f]) 
}
means <- data.frame(feature, clusters, values)
means$clusters <- as.factor(means$clusters)

ggplot(means, aes(fill=clusters, y=values, x=feature),) +
  geom_bar(position=position_dodge(width = .5), width = 1.5 ,stat="identity", alpha=.8)+ 
  labs(title='Comparing standardized returns\'\n means among clusters') +
  theme_fivethirtyeight() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0, hjust=1), 
        legend.position = "top") +
  scale_fill_brewer(palette = "Set1") 

## Neural Network ####

returns$USD.CNY <- NULL
returns$USD.EUR <- NULL
returns$USD.JPY <- NULL
returns$XRP <- NULL
# returns$OIL <- NULL
# returns$AMD <- NULL
# returns$PYPL <- NULL
# returns$INTC <- NULL
returns$GOLD <- NULL
returns$ETH <- NULL
returns$BTC <- NULL
returns$LTC <- NULL

train <- returns[returns$Date <= as.Date("2018-12-31"), -1]
validation <- returns[(returns$Date > as.Date("2018-12-31") & returns$Date <= as.Date("2019-12-31")), -1]
test <- returns[returns$Date > as.Date("2019-12-31"), -1]

formula <- as.formula(NASDAQ ~ .)

width <- c(2, 5, 7, 10, 12, 15, 20, 25, 30)

to_be_predicted <- 'NASDAQ'
threshold <- 0.05
maxsteps <- 150000

valid_scores <- as.data.frame(matrix(ncol = 6, nrow = 200))
colnames(valid_scores) <- c("train_rmse", "valid_rmse", "width_L1", "width_L2", "width_L3", "width_L4")
offset <- 0

set.seed(999)
## depth = 1 ####
ite <- 0
for (j in 1:length(width)) {
    ite <- ite + 1
    net <- neuralnet(formula, 
                    data = train, 
                    hidden = c(width[j]),
                    linear.output = TRUE,
                    threshold = threshold,
                    stepmax = maxsteps,
                    err.fct = "sse",
                    lifesign = "minimal")
    
    nn_pred <- predict(net, train)
    sse <-  (train[, colnames(train) == to_be_predicted] - nn_pred)^2
    valid_scores$train_rmse[offset + ite] <- sqrt(mean(sse))
    
    nn_pred <- predict(net, validation)
    sse <-  (validation[, colnames(validation) == to_be_predicted] - nn_pred)^2
    valid_scores$valid_rmse[offset + ite] <- sqrt(mean(sse))
    valid_scores$width_L1[offset + ite] <- width[j]
    valid_scores$width_L2[offset + ite] <- 0
    valid_scores$width_L3[offset + ite] <- 0
    valid_scores$width_L4[offset + ite] <- 0
}

offset <- ite + offset 
## depth = 2 ####
for (i in 1:length(width)) {
  for (j in 1:length(width)){
    ite <-  0
    if (width[j] >= width[i]) {
      ite <- ite + 1
      net <- neuralnet(formula, 
                       data = train, 
                       hidden = c(width[j], width[i]), 
                       linear.output = TRUE, 
                       threshold = threshold,
                       stepmax = maxsteps,
                       err.fct = "sse",
                       lifesign = "minimal")
      
      nn_pred <- predict(net, train)
      sse <-  (train[, colnames(train) == to_be_predicted] - nn_pred)^2
      valid_scores$train_rmse[offset + ite] <- sqrt(mean(sse))
      
      nn_pred <- predict(net, validation)
      sse <-  (validation[, colnames(validation) == to_be_predicted] - nn_pred)^2
      valid_scores$valid_rmse[offset + ite] <- sqrt(mean(sse))
      valid_scores$width_L1[offset + ite] <- width[j]
      valid_scores$width_L2[offset + ite] <- width[i]
      valid_scores$width_L3[offset + ite] <- 0
      valid_scores$width_L4[offset + ite] <- 0
    }
    offset <- offset + ite
  }
}
## depth = 3 ####
for (k in 1:length(width)) {
  for (i in 1:length(width)) {
    for (j in 1:length(width)) {
      ite <- 0
      if (((width[k] <= width[i]) & (width[i]) <= width[j]) & (nrow(train)) > width[k]*width[i]*width[j]){
        ite <- ite + 1
        net <- neuralnet(formula, 
                         data = train, 
                         hidden = c(width[j], width[i], width[k]), 
                         linear.output = TRUE, 
                         threshold = threshold,
                         stepmax = maxsteps,
                         err.fct = "sse",
                         lifesign = "minimal")
        
        nn_pred <- predict(net, train)
        sse <-  (train[, colnames(train) == to_be_predicted] - nn_pred)^2
        valid_scores$train_rmse[offset + ite] <- sqrt(mean(sse))
        
        nn_pred <- predict(net, validation)
        sse <-  (validation[, colnames(validation) == to_be_predicted] - nn_pred)^2
        valid_scores$valid_rmse[offset + ite] <- sqrt(mean(sse))
        valid_scores$width_L1[offset + ite] <- width[j]
        valid_scores$width_L2[offset + ite] <- width[i]
        valid_scores$width_L3[offset + ite] <- width[k]
        valid_scores$width_L4[offset + ite] <- 0
      }
      offset <- offset + ite
    }
  }
}
## depth = 4 ####
for (z in 1:length(width)) {
  for (k in 1:length(width)) {
    for (i in 1:length(width)) {
      for (j in 1:length(width)) {
        ite <- 0
        if (((width[z] <= width[k]) & (width[k] <= width[i]) & (width[i]) <= width[j]) & (nrow(train)) > width[z]*width[k]*width[i]*width[j]){
          ite <- ite + 1
          net <- neuralnet(formula, 
                         data = train, 
                         hidden = c(width[j], width[i], width[k], width[z]), 
                         linear.output = TRUE, 
                         threshold = threshold,
                         stepmax = maxsteps,
                         err.fct = "sse",
                         lifesign = "minimal")
          nn_pred <- predict(net, train)
          #nn_pred <- nn_pred$net.result
          sse <-  (train[, colnames(train) == to_be_predicted] - nn_pred)^2
          valid_scores$train_rmse[offset + ite] <- sqrt(mean(sse))
        
          nn_pred <- predict(net, validation)
          #nn_pred <- nn_pred$net.result
          sse <-  (validation[, colnames(validation) == to_be_predicted] - nn_pred)^2
          valid_scores$valid_rmse[offset + ite] <- sqrt(mean(sse))
          valid_scores$width_L1[offset + ite] <- width[j]
          valid_scores$width_L2[offset + ite] <- width[i]
          valid_scores$width_L3[offset + ite] <- width[k]
          valid_scores$width_L4[offset + ite] <- width[z]
      }
        offset <- offset + ite
    }
    }
  }
}
## best NN testing####
valid_scores <- na.omit(valid_scores)
best_arc <- valid_scores[valid_scores$valid_rmse == min(valid_scores$valid_rmse), ]
layers <- NULL
for (j in 3:ncol(best_arc)) {
  if(best_arc[,j] != 0) {
    layers <- c(layers, best_arc[,j])
  }
}
train <- returns[returns$Date <= as.Date("2019-12-31"), -1]
best_net <- neuralnet(formula = formula,
                      data = train, 
                      err.fct = "sse", 
                      hidden = layers,
                      linear.output = TRUE, 
                      threshold = threshold,
                      lifesign = "minimal")

# plot(best_net, intercept = F, show.weights = F)
plot(best_net, intercept = F, show.weights = F,arrow.length = 0.15,
    col.entry.synapse = "black", 
    col.entry = "darkred", col.hidden = "darkblue", 
    col.hidden.synapse = "darkred", col.out = "darkred", 
    col.out.synapse = "darkgreen")

nn_pred <- predict(best_net, test)
sse <-  (test[, colnames(test) == to_be_predicted] - nn_pred)^2
test_rmse <- sqrt(mean(sse))
test_rmse

x <- returns[, which(names(returns) == to_be_predicted)]
# x <- ifelse(x < -0.25, 0, x)

ggplot(data = returns) + 
  geom_boxplot(aes(y = as.factor(to_be_predicted),x =x) ) +
  geom_point(aes(y = as.factor(to_be_predicted), x=test_rmse),
             color = "red", size = 3.5) +
  annotate("text", vjust = -.5, y = as.factor(to_be_predicted),
           x = test_rmse, label="Test RMSE", 
           color = "red", size = 10,) +
  labs(x="", y = 'BTC returns',
       color = 'Test RMSE')+
  theme_fivethirtyeight() + 
  theme(axis.title.x = element_text(),
        axis.title.y = element_blank(),
        axis.text.y = element_text(size = 12),
        legend.position = "top",
        legend.text = element_text())

## trying to compute & plot Shapley Values ####
library("iml")
X <- train[which(names(test) != to_be_predicted)]
y <- train[which(names(test) == to_be_predicted)]
predictor <- Predictor$new(best_net, data = X, y = y)

shapley <- Shapley$new(predictor, x.interest = X[1:nrow(X)/2, ])

plot(shapley)

# imp <- FeatureImp$new(predictor, loss = "mse")
# plot(imp)

## Google Trends ####

gtrend <- read.csv('./GoogleTrendsBTC.csv')
gtrend$Date <- as.Date(gtrend$Date)
crypto_gTrend <- merge(x=crypto[,1:2], y=gtrend, all.x = TRUE)
crypto_gTrend[2:3] <- as.data.frame(sapply(crypto_gTrend[2:3], function(x) nafill(x, type = 'nocb')))
# the last date has no more values available forward on, hence the na value is 
# replaced with the last non-na available value
crypto_gTrend[2:3] <- as.data.frame(sapply(crypto_gTrend[2:3], function(x) nafill(x, type = 'locf')))

ggplot(crypto_gTrend, aes(Date))+
  geom_line(aes(y=BTC/max(BTC)*100, colour='BTC (scaled down)'), alpha=.75)+
  geom_line(aes(y=GtrendBTC, colour='Google Trend BTC'), alpha = .6)+
  labs(y='%', x='Date',
       color='Index: ', 
       title = "Bitcoin google research trend in time",
       subtitle = 'From 1-1-2016 to 29-12-2020')+
  theme_fivethirtyeight()+
  theme(axis.title = element_text())+
  scale_color_brewer(palette = 'Dark2')

crypto_gTrend[crypto_gTrend$GtrendBTC == max(crypto_gTrend$GtrendBTC),]
