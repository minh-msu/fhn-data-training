from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

def tuning(model, X_train, y_train):
    cv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        "n_estimators": [100, 300, 500],          # number of boosting rounds
        "max_depth": [3, 5, 7, 9],                # tree depth
        "learning_rate": [0.01, 0.05, 0.1, 0.2],  # shrinkage step size
    }
    #     "subsample": [0.6, 0.8, 1.0],             # row sampling
    #     "colsample_bytree": [0.6, 0.8, 1.0],      # feature sampling
    #     "min_child_weight": [1, 3, 5],            # minimum sum of instance weight (hessian) needed in a child
    #     "gamma": [0, 0.1, 0.2, 0.5],              # minimum loss reduction required for a split
    #     "reg_alpha": [0, 0.1, 1],                 # L1 regularization
    #     "reg_lambda": [1, 5, 10]                  # L2 regularization
    # }

    
    grid = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring="neg_root_mean_squared_error",
        cv=cv, 
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_
