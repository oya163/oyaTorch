
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

N, D_in, H, D_out = 64, 1000, 100, 10


# In[3]:

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)


# In[4]:

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)


# In[5]:

learning_rate = 1e-6


# In[6]:

for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)
    
    
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    
    w1 = w1 - learning_rate * grad_w1
    w2 = w2 - learning_rate * grad_w2
    


# In[ ]:



