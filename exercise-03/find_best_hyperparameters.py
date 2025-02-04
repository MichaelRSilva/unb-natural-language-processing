from sklearn.model_selection import GridSearchCV

def find_best_hyperparameters(param_grid, pipeline, x_train, y_train, num_classes):
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=num_classes, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_param = grid_search.best_params_

    return best_model, best_param

