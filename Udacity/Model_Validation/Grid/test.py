from sklearn.model_selection import GridSearchCV
parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}


from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
scorer = make_scorer(f1_score)

# Create the object.
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
# Fit the data
grid_fit = grid_obj.fit(X, y)


best_clf = grid_fit.best_estimator_