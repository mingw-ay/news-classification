# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
# from sklearn.pipeline import Pipeline

# steps = [('dense', FunctionTransformer(
#     func=lambda X: X.toarray(), accept_sparse=True)),
#     ('model', None)
# ]
# pipe = Pipeline(steps=steps)
# param = {
#     'model': [GaussianNB(), BernoulliNB(), MultinomialNB(), ComplementNB()]
# }
# gs = GridSearchCV(estimator=pipe, param_grid=param, cv=2,
#                   scoring='f1', n_jobs=1, verbose=10)
# gs.fit(X_train_tran, y_train)
# print('best_grams:', gs.best_params_)
# print(gs.best_estimator_)
