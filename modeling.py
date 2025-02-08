import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode(rule : str):
    a = np.load("encoding.npy")
    voc = {}
    for i in range(0,len(rule)-1):
        voc[rule[i]] = a[i]
    return voc

def tensorfy(s: str, voc: dict):
    x = np.array([voc[s[0]], voc[s[1]]])
    
    for i in range(2, len(s)):
        new_element = np.array([voc[s[i]]])
        x = np.append(x, new_element, axis=0) 
    return x

def super_tensorfy(s : list, voc: dict):
    y = np.array([tensorfy(s[0], voc)])
    for i in range(1,len(s)-1):
        np.append(y,tensorfy(s[i], voc))
    return y

voc = encode("""ABCDEFGHIJKLMNOPQRSTUVWXYZ
       abcdefghijklmnopqrstuvwxyz0123456789
       .,:;?!"'()[]{}
       -–—.../_+−×÷/=≠><≥≤~%‰&|^$€£¥¢@#°§¶®™©• """)
t = pd.read_pickle("final.pkl")
a = tensorfy(t.loc[0,"wrong"], voc)
b = tensorfy(t.loc[0,"correct"], voc)
c = tensorfy(t.loc[1,"wrong"], voc)
d = tensorfy(t.loc[1,"correct"], voc)
f = np.array([[a,b],[c,d]])
print(f)







