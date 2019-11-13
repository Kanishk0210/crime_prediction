import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

data=pd.read_excel("/home/kanishk/Desktop/crimedata.xlsx")
df=pd.DataFrame(data)
#print(df)
x=np.asanyarray(df['y'])
x_train=x[:15,np.newaxis]
#print(x_train)
y=np.asanyarray(df['Total'])
y_train=y[:15,np.newaxis]
#print(y_train)
m=LinearRegression().fit(x_train,y_train)
yfit=m.predict(x_train)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,yfit,linewidth=5,alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Total Crimes')
plt.legend(['prediction','actual'])
plt.show()
print("score : ",m.score(x_train,y_train))
print("rms : ",mean_squared_error(x_train,yfit))
print("r2 score : ",r2_score(x_train,yfit))
