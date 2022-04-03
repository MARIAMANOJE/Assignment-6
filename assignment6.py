#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# In[3]:


hw=pd.read_csv('https://gist.githubusercontent.com/nstokoe/7d4717e96c21b8ad04ec91f361b000cb/raw/bf95a2e30fceb9f2ae990eac8379fc7d844a0196/weight-height.csv')


# In[4]:


hw.head()


# In[5]:


hw.describe()


# In[6]:


plt.hist(hw.Height)


# In[7]:


from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


# In[8]:


stats.shapiro(hw.Height)


# In[9]:


stats.shapiro(hw.Weight)


# In[10]:


hw.shape


# In[11]:


train=hw.iloc[:8000]


# In[12]:


test=hw.iloc[8000:]


# In[13]:


train.shape


# In[14]:


test.shape


# In[27]:


sns.scatterplot(x=hw.Height,y=hw.Weight,data=hw,color={'Height':'yellow','Weight':'green'})


# In[21]:


sns.lineplot(hw.Height,hw.Weight)


# In[17]:


from scipy.stats import pearsonr


# In[18]:


corr, _ = pearsonr(hw.Height,hw.Weight)


# In[21]:


corr


# In[22]:


_


# In[23]:


import statsmodels.api as sm


# In[24]:


train_x=train.Weight


# In[ ]:


test_y=train.Height


# In[ ]:


train_x=sm.add_constant(train_x)


# In[ ]:


train_x


# In[ ]:


model=sm.OLS(train_y,train_x).fit()


# In[ ]:


model.summary()


# In[ ]:


model.params


# In[ ]:


test_x=test.Weight


# In[ ]:


test_y=test.Height


# In[ ]:


test_x=sm.add_constant(test_x)


# In[ ]:


test_x


# In[ ]:


predict=model.predict(test_x)


# In[ ]:


predict


# In[ ]:


x=np.array(hw.Weight,hw.Height)
y=np.array(hw.Weight,hw.Height)


# In[ ]:


sns.countplot(hw.Height,data=hw)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




