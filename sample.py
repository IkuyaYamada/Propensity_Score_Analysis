# calculate propensity score
# with rerugalization 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.linear_model import LogisticRegression 
from sklearn import tree

data = pd.read_csv("/Users/yamadaikuya/Desktop/Research/interim_presentation_2019/codes/pruned_ds_0510.csv")
#data.astype("float64")

data = data.dropna(subset=['Age', 'female', 'Years_of_Schooling', 'Math_Score',
       'parents_are_farmers', 'born_in_this_village', 'Risk_averse',
       'Competitive', 'Absolute_Overconfidence', 'Relative_Overconfidence', "Cut_Flower"])
X = data[['Age', 'female', 'Years_of_Schooling', 'Math_Score',
       'parents_are_farmers', 'born_in_this_village', 'Risk_averse',
       'Competitive', 'Absolute_Overconfidence', 'Relative_Overconfidence']]
Y = data[["Cut_Flower"]].values.ravel()
print(Y.shape)
# logit model as ordinary
def logit_model(X, Y):
	clf = LogisticRegression(random_state=0, solver="liblinear",).fit(X, Y)
	ps = clf.predict_proba(X)
	return ps
ps = logit_model(X, Y)
print(ps[0:5, :])
print(len(ps))

pd.DataFrame(ps).to_csv("logit.csv")
