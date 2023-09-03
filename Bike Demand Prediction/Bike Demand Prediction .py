#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# In[2]:


df=pd.read_csv("C:/Users/Samarth Thakur/Downloads/hour (1).csv")
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().any()#checking any null values


# In[7]:


#now we have to drop certain column for better accuracy of our model
bikes_df=df.copy()
bikes_df=bikes_df.drop(['index','date','casual','registered'] , axis=1)


# In[8]:


bikes_df.shape


# here you can see our column reduces from 17 to 13

# In[9]:


bikes_df.head()


# # Visualize the data

# In[10]:


import matplotlib.pyplot as plt
bikes_df.hist(rwidth=0.9)
plt.tight_layout()


# In[11]:


plt.subplot(2,2,1)
plt.title('Temperature  Vs  Demand')
plt.scatter(bikes_df['temp'],bikes_df['demand'],s=1)

plt.subplot(2,2,2)
plt.title('aTemp Vs Demand')
plt.scatter(bikes_df['atemp'],bikes_df['demand'],s=1,c='y')

plt.subplot(2,2,3)
plt.title('Humidity Vs Demand')
plt.scatter(bikes_df['humidity'],bikes_df['demand'],s=0.5,c='b')

plt.subplot(2,2,4)
plt.title('Windspeed Vs Demand')
plt.scatter(bikes_df['windspeed'],bikes_df['demand'],s=0.77,c='r')

plt.tight_layout()
plt.show()


# In[12]:


#plot categoricalfeatures vs demand
plt.subplot(3,3,1)
plt.title("Avg demand per season")
#create a list of unique season's values
cat_list=bikes_df['season'].unique()
#create avg demand per season using groupby
cat_avg = bikes_df.groupby('season')['demand'].mean()
plt.bar(cat_list,cat_avg)



plt.subplot(3,3,2)
plt.title("Avg demand per month")
cat_list=bikes_df['month'].unique()
cat_avg = bikes_df.groupby('month')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.subplot(3,3,3)
plt.title("Avg demand per weather")
cat_list=bikes_df['weather'].unique()
cat_avg = bikes_df.groupby('weather')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.subplot(3,3,4)
plt.title("Avg demand per year")
cat_list=bikes_df['year'].unique()
cat_avg = bikes_df.groupby('year')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.subplot(3,3,5)
plt.title("Avg demand per holiday")
cat_list=bikes_df['holiday'].unique()
cat_avg = bikes_df.groupby('holiday')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.subplot(3,3,6)
plt.title("Avg demand per hour")
cat_list=bikes_df['hour'].unique()
cat_avg = bikes_df.groupby('hour')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.subplot(3,3,7 )
plt.title("Avg demand per workingday")
cat_list=bikes_df['workingday'].unique()
cat_avg = bikes_df.groupby('workingday')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.subplot(3,3,8 )
plt.title("Avg demand per weekday")
cat_list=bikes_df['weekday'].unique()
cat_avg = bikes_df.groupby('weekday')['demand'].mean()
plt.bar(cat_list,cat_avg)

plt.tight_layout()
plt.show



# now we drop some feature like
# -weekday because it hardly shows any variation
# -year because it is of only 2 years and we dont know 5 years down the lane
# -workingday same it does not show any change

# In[13]:


plt.title("Avg demand per hour")
cat_list=bikes_df['hour'].unique()
cat_avg = bikes_df.groupby('hour')['demand'].mean()
plt.bar(cat_list,cat_avg,color=['r','g','c','y',])


# In[14]:


#checking outliers
bikes_df['demand'].describe()


# here you can see min is 1 max is 977 and 50% that is fifty percent of data is fairly away from max and mean now we can check at different quantile

# In[15]:


bikes_df['demand'].quantile([0.05,0.10,0.15,0.9,0.95,0.99])


# from above we can say that 5% of time the demand is less than or equal to 5
# similary only 1% of time it is greater than or equal to 782 so these two are outliers

# ## Check the multiple linear regression assumption

# In[16]:


#linearity using correlation matrix
correl=bikes_df[['temp','atemp','humidity','windspeed','demand']].corr()
print(correl)


# Multicollinearity is a term used in statistics and regression analysis to describe a situation in which two or more independent variables (also known as predictor variables or features) in a regression model are highly correlated with each other.
# 
# Multicollinearity occurs when there is a high correlation between two or more independent variables. This high correlation can make it difficult to determine the individual effect of each independent variable on the dependent variable in a regression model.
# 
#     
# Effects on Regression Analysis: Multicollinearity can have several adverse effects on regression analysis, including:
# 
#     It can make it challenging to identify the true relationship between         independent variables and the dependent variable.
#     
#     If multicollinearity is detected, there are several strategies to address it:
# 
# Removing one or more of the highly correlated independent variables.
# Combining correlated variables into a single variable.

# In[17]:


#we drop some  feature from above all cell findings
bikes_df=bikes_df.drop(['weekday','year','workingday','atemp','windspeed'],axis=1)


# In[18]:


bikes_df.shape


# In[19]:


bikes_df1=pd.to_numeric(bikes_df['demand'],downcast='float')
plt.acorr(bikes_df1,maxlags=12)


# In[20]:


df1=bikes_df['demand']
df2=np.log(df1)

plt.figure()
df1.hist(rwidth=0.9,bins=20,color='r')

plt.figure()
df2.hist(rwidth=0.9,bins=20)

bikes_df['demand']=np.log(bikes_df['demand'])


# The normality assumption in regression, specifically in linear regression analysis, is one of the key assumptions that should hold for the model to be valid. This assumption pertains to the distribution of the residuals (the differences between the observed values and the predicted values) and is often referred to as the "normality of residuals" or the "normality of errors."
# 
# The normality assumption states that the residuals or errors should be normally distributed, which means they should follow a normal or Gaussian distribution. In a normal distribution:
# 
#  ->   The data is symmetrically distributed around the mean.
#  -> The majority of the data points cluster around the mean.

# In[21]:


print(bikes_df)


# In[22]:


t_1=bikes_df['demand'].shift(+1).to_frame()
t_1.columns=['t-1']

t_2=bikes_df['demand'].shift(+2).to_frame()
t_2.columns=['t-2']

t_3=bikes_df['demand'].shift(+3).to_frame()
t_3.columns=['t-3']

bikes_df_lag=pd.concat([bikes_df,t_1,t_2,t_3],axis=1)
#3 more columns added

print(bikes_df_lag)


# In[23]:


bikes_df_lag=bikes_df_lag.dropna()
bikes_df_lag #null values drop


# In[24]:


bikes_df_lag.dtypes


# we have to convert int to categorical for dummy variable 

# In[25]:


# Assuming you have a DataFrame named 'bikes_df_lag'

# Convert categorical variables to the 'category' data type
bikes_df_lag['season'] = bikes_df_lag['season'].astype('category')
bikes_df_lag['holiday'] = bikes_df_lag['holiday'].astype('category')
bikes_df_lag['weather'] = bikes_df_lag['weather'].astype('category')
bikes_df_lag['month'] = bikes_df_lag['month'].astype('category')
bikes_df_lag['hour'] = bikes_df_lag['hour'].astype('category')

# Perform one-hot encoding using get_dummies and drop the first category (to avoid multicollinearity)
bikes_df_lag = pd.get_dummies(bikes_df_lag, drop_first=True)

bikes_df_lag=pd.get_dummies(bikes_df_lag,drop_first=True)
bikes_df_lag


# In[ ]:





# ## Train & Test (Model Training)

# most important our data is time series data so we have to be careful so they that it doesn't loose autocollinerity

# In[26]:


# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=\
#     train_test_split(X,Y,test_size=0.4,random_state=1234)


# In[27]:


Y=bikes_df_lag[['demand']]
X=bikes_df_lag.drop(['demand'],axis=1)


# In[32]:


#create the size for 70% of the data
tr_size=0.7*len(X) 
tr_size=int(tr_size)

#create the train and test using the tr_size
X_train=X.values[0:tr_size]
X_test=X.values[tr_size:len(X)]

Y_train=Y.values[0:tr_size]
Y_test=Y.values[tr_size:len(Y)]


# ## Fit and Score the model

# In[33]:


#Linear Regression
from sklearn.linear_model import LinearRegression

std_reg=LinearRegression()
std_reg.fit(X_train,Y_train)

r2_train=std_reg.score(X_train,Y_train)
r2_test=std_reg.score(X_test,Y_test)

Y_predict=std_reg.predict(X_test)


# In[34]:


from sklearn.metrics import mean_squared_error 

rmse=math.sqrt(mean_squared_error(Y_test,Y_predict))

print(rmse)


# In[35]:


# Assuming Y_test and Y_predict are NumPy arrays
Y_test = Y_test.reshape(-1)  # Ensure 1-dimensional
Y_predict = Y_predict.reshape(-1)

# Convert the true and predicted target values from log-scale to original scale
Y_test_e = [math.exp(val) for val in Y_test]
Y_predict_e = [math.exp(val) for val in Y_predict]

# Initialize a variable to store the sum of squared logarithmic differences
log_sq_sum = 0

# Calculate the squared logarithmic differences for each data point and sum them
for i in range(len(Y_test_e)):
    log_a = math.log(Y_test_e[i] + 1)  # Adding 1 to avoid logarithm of zero
    log_p = math.log(Y_predict_e[i] + 1)
    log_diff = (log_p - log_a) ** 2
    log_sq_sum += log_diff

# Calculate the RMSLE
rmsle = math.sqrt(log_sq_sum / len(Y_test))
print(rmsle)


# In[ ]:




