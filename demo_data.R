#library(openxlsx)
n<-200
f_sphere<-function(x){
  out<-matrix(0,length(x),3)
  for (i in 1:length(x)) {
    out[i,]<-c(sin(abs(pi*x[i]/4)),cos(abs(pi*x[i]/4)),0)
  }
  return(out)
}
set.seed(1)
beta0<-c(1,0,0,0)
Training_x<-list();Training_y<-list()
for (i in 1:n) {
  Training_x[[i]]<-c(runif(1,-1,1),runif(1,-1,1),runif(1,-1,1),runif(1,-1,1))
  # mu<-c(1,0,0)
  # V<-c(0,rnorm(2,0,0.2))
  mu<-as.vector(f_sphere(as.numeric(Training_x[[i]]%*%beta0))) 
  V<-c(0,0,rnorm(1,0,0.5))
  Training_y[[i]]<-cos(norm(V,type = "2"))*mu+
    sin(norm(V,type = "2"))*V/norm(V,type = "2")
}
df_x <- do.call(rbind, Training_x)
df_y <- do.call(rbind, Training_y)
colnames(df_x)<-c("x1","x2","x3","x4")
colnames(df_y)<-c("y1","y2","y3")
df<-data.frame(cbind(df_x,df_y)) 
write.csv(df, "data_demo.csv")
#write.xlsx(df, "data_demo.xlsx")




