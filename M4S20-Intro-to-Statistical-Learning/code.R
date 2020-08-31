library(tidyverse)
library(ggplot2)
library(MASS)
library(smacof)
library(factoextra)
library(NbClust)

# Loading data and scaling
df = read.csv("eu_data_economical.csv")
rownames(df) = df$country
df = subset(df, select=-country)
df = scale(df)
n = 27
p = 38

###  MULTIDIMENSIONAL SCALING
euclidean_distance = dist(df, method="euclidean")
euclidean_distance_obj = as.dist(euclidean_distance)

scaling_solution_euclidean = cmdscale(euclidean_distance_obj, k=2, eig=TRUE)

# Euclidean Eigenvalue Plot
the.eigs.euclid = scaling_solution_euclidean$eig
fig01 = ggplot()
fig01 + geom_point(aes(1:n, scaling_solution_euclidean$eig)) +
    ggtitle("Figure 01: Eigenvalues for Multi-Dimensional Scaling") +
    xlab("Eigenvalue Number") +
    ylab("Eigenvalue Value")
ggsave("figures/fig01.png", width=10, height=7)

jpeg("figures/fig02.png", width=500, height =500)
plot(scaling_solution_euclidean$points[,1],
     scaling_solution_euclidean$points[,2],
     main="Figure 02: Recovered Configuration Multi-Dimensional Scaling",
     xlab="X Coordinate", ylab="Y Coordinate",
     type="n")
text(scaling_solution_euclidean$points[,1],
     scaling_solution_euclidean$points[,2],
     labels=dimnames(df)[[1]],
     cex=0.7)
dev.off()

### ORDINAL SCALING - MANHATTAN DISTANCE
manhattan_distance = dist(df, method="manhattan")
manhattan_distance_obj = as.dist(manhattan_distance)

scaling_solution_manhattan = cmdscale(manhattan_distance_obj, k=2, eig=TRUE)
scaling_solution_manhattan_isoMDS = isoMDS(manhattan_distance_obj)

# Manhattan Eigenvalue Plots
fig03 = ggplot()
fig03 + geom_point(aes(1:n, scaling_solution_manhattan$eig)) +
    ggtitle("Figure 03: Eigenvalues for MultiDimensional Scaling Manhattan") +
    xlab("Eigenvalue Number") +
    ylab("Eigenvalue Value") +
    geom_vline(xintercept=20, linetype='dashed')
ggsave("figures/fig03.png", width=10, height=7)

# Plot of Eigenvalues
jpeg("figures/fig04.png", width = 500, height = 500)
the.eigs = scaling_solution_manhattan$eig
the.eigs[20] = 0
plot(1:27, log(abs(the.eigs)),
     xlab="Eigenvalue number",
     ylab="Log(Abs(Eigenvalue))",
     main="Figure 04: log(|Eigenvalues|) for Manhattan Distance",
     type="n")
points(1:19, log(abs(the.eigs[1:19])), col=1)
points(21:27, log(abs(the.eigs[21:27])), col=4)
abline(h=0, lty=2)
abline(v=20, lty=2, col=2)
text(x=22, y=12.75, label="Zero eigenvalue")
legend(x="bottomleft", col=c(1,4), pch=1, legend=c("Positive", "Negative"))
dev.off()

# Manhattan Configuration
jpeg("figures/fig05.png", width = 500, height = 500)
plot(scaling_solution_manhattan_isoMDS$points[,1],
     scaling_solution_manhattan_isoMDS$points[,2],
     main="Figure 05: Configuration using Manhattan Distances",
     xlab="X Coordinate", ylab="Y Coordinate",
     type="n")
text(scaling_solution_manhattan_isoMDS$points[,1],
     scaling_solution_manhattan_isoMDS$points[,2],
     labels=dimnames(df)[[1]],
     cex=0.7)
dev.off()

# PROCRUSTES ANALYSIS
sol1 = scaling_solution_euclidean$points
sol2 = scaling_solution_manhattan_isoMDS$points

X = cbind(as.numeric(sol1[,1]), as.numeric(sol1[,2]))
Xstar = cbind(as.numeric(sol2[,1]), as.numeric(sol2[,2]))

scale_procr = Procrustes(X=X, Y=Xstar)
Yhat = scale_procr$Yhat

jpeg("figures/fig06.png", width = 500, height = 500)
plot(sol1[,1], sol1[,2],
     main="Figure 06: Plot of Both Configurations Superimposed",
     xlab="X Coordinate", ylab="Y Coordinate",
     type="n")
text(sol1[,1], sol1[,2],
     labels=dimnames(df)[[1]],
     cex=0.7)
text(Yhat[,1], Yhat[,2],
    labels=dimnames(df)[[1]],
    cex=0.7, col=4)
dev.off()

# MASTERY: K-MEANS
fviz_nbclust(sol1, kmeans, method="silhouette")
ggsave("figures/fig07.png", width=10, height=7)
fviz_nbclust(sol1, kmeans, method="wss")
ggsave("figures/fig08.png", width=10, height=7)

kmean_sol = kmeans(x=sol1, centers=3)
jpeg("figures/fig09.png", width=500, height=500)
plot(sol1[,1], sol1[,2], type="n",
    main="Figure 09: K-Means Clustering K=3",
    xlab="X Coordinate", ylab="Y Coordinate")
text(sol1[,1], sol1[,2], col=kmean_sol$cluster, lab=dimnames(df)[[1]])
dev.off()
