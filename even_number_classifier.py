#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import matplotlib as mpl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[2]:


import numpy as np
import os


# In[3]:


np.random.seed(42)


# In[4]:


# To plot pretty figures
# %matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[5]:


# Where to save the figures
PROJECT_ROOT_DIR = "."


# In[6]:


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


# In[7]:


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    # fetch_openml() returns targets as strings
    mnist.target = mnist.target.astype(np.int8)
    sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
mnist["data"], mnist["target"]


# In[8]:


mnist.data.shape


# In[9]:



X, y = mnist["data"], mnist["target"]
X.shape


# In[10]:



y.shape


# In[11]:


some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()


# In[12]:


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


# In[13]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
from sklearn.linear_model import SGDClassifier
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train


# In[14]:


y_train_even = (y_train%2==0)
y_test_even = (y_test%2==0)


# In[15]:


y_train_even


# In[16]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_even)


# In[17]:


some_digit = X[28000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
sgd_clf.predict([some_digit])


# In[18]:


some_digit = X[31000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
sgd_clf.predict([some_digit])


# In[19]:


y_pred = sgd_clf.predict(X_test)


# In[20]:


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_even, cv=3, scoring="accuracy")


# In[21]:


from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_even, cv=3)
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_even, y_train_pred)


# In[25]:


from sklearn.metrics import f1_score, accuracy_score
f1_score(y_train_even, y_train_pred)


# In[27]:


accuracy_score(y_train_even, y_train_pred)


# In[26]:


recall_score(y_train_even, y_train_pred)


# # KNeighborsClassifier

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train, y_train_even)


# In[29]:


some_digit = X[28000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
kn_clf.predict([some_digit])


# In[30]:


some_digit = X[31000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")

plt.show()
kn_clf.predict([some_digit])


# In[40]:


y_kn_pred = cross_val_predict(kn_clf, X_train[:100,:], y_train_even[:100], cv=3)


# In[42]:


accuracy_score(y_train_even[:100], y_kn_pred)


# In[ ]:




