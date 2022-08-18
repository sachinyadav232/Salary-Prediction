#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error


# In[12]:


#Load Data
import pandas as pd
data = pd.read_csv('Salary.csv')
data.head()


# In[13]:


#Perform EDA
#Check null value is present or not.
data.isnull().sum()


# In[14]:


data.info()


# In[15]:


data.describe()


# In[16]:


#Visualize Data
import matplotlib.pyplot as plt
plt.scatter( data['YearsExperience'] ,data['Salary'] )
plt.xlabel('Year of Exp' )
plt.ylabel('Salary' )
plt.show()


# In[17]:


#Prepare Data
X = data.drop( 'Salary', axis=1)
y = data['Salary']


# In[18]:


X.shape , y.shape


# In[21]:


#Split data  into train and test.
from sklearn.model_selection import train_test_split 
X_train , X_test , Y_train , Y_test = train_test_split(X,y,random_state=101,test_size=0.2)
X_train.shape , X_test.shape , Y_train.shape , Y_test.shape


# In[22]:


#Define Linear regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)


# In[23]:


#Test model
pred = lr.predict(X_test)
pred


# In[24]:


Y_test


# In[25]:


#Check Actual data,Predicated Data and difference between the Actual and Predicated Data.
import numpy as np
diff = Y_test - pred
pd.DataFrame(np.c_[Y_test,pred,diff] , columns=['Actual','Predicted','Difference'])


# In[26]:


#Visualize Model,that how it is performing on training data.
plt.scatter(X_train , Y_train , color='blue')
plt.plot(X_train ,lr.predict(X_train),color='red')
plt.title('Salary vs Experience')
plt.ylabel('Salary')
plt.show()


# In[27]:


#Visualize Model, that how it is performing on testing data.
plt.scatter(X_test , Y_test ,color='blue')
plt.plot(X_test , lr.predict(X_test) ,color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[28]:


#Evaluate
lr.score(X_test , Y_test)


# In[29]:


# from sklearn.metrics import r2_score , mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test,pred))
r2 = r2_score(Y_test,pred)
rmse , r2


# In[30]:


# Test on the custom data.
exp = 3
lr.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(lr.predict([[exp]])[0])} thousands")


# In[31]:


exp = 5
lr.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(lr.predict([[exp]])[0])} thousands")


# In[32]:


exp = 9
lr.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(lr.predict([[exp]])[0])} thousands")


# In[ ]:




