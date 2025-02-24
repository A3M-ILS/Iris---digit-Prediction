#!/usr/bin/env python
# coding: utf-8

# In[167]:


import pandas as pd 
from sklearn.datasets import load_digits 
digits = load_digits()


# In[168]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.gray() 
for i in range(4):
    plt.matshow(digits.images[i])


# In[169]:


df = pd.DataFrame(digits.data)
df.head()


# In[170]:


df['target'] = digits.target
df[0:12]


# In[171]:


X = df.drop('target',axis='columns')
y = df.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=70)
model.fit(X_train, y_train)


# In[172]:


model.score(X_test,y_test)


# In[173]:


y_predicted = model.predict(X_test)


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_predicted)
CM


# In[174]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(12,8))
sn.heatmap(CM, annot=True)
plt.xlabel('Predicted_values')
plt.ylabel('True_values')

