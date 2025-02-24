#!/usr/bin/env python
# coding: utf-8

# In[42]:


from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
dir(iris)


# In[43]:


df = pd.DataFrame(iris.data)
df["target"]=iris.target
df.head(10)


# In[44]:


X = df.drop("target",axis="columns")
Y = df.target
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=15)
model.fit(X_train, Y_train)
model.score(X_test,Y_test)


# In[45]:


Y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(Y_test,Y_predicted)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(CM,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

