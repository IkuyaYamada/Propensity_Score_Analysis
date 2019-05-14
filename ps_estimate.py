# calculate propensity score
# with rerugalization 
import numpy as np 
import matplotlib.pyplot as plt
from  sklearn.liner_model import LogisticRegression 
from sklearn import tree

# logit model as ordinary
def logit_model(X, Y):
	clf = LogisticRegression(random_state=0, solver="liblinear",).fit(X, Y)
	ps = clf.predict_proba(X_)
	return ps


# creat interaction term as much as possible 
def creat_interaction(*beta):
	beta = []


# create higher order term
def create_higher_order():



# def 