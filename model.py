from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def train_with_grid_search(X_train_scaled, y_train, X_test_scaled, y_test):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'class_weight': ['balanced', None]
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validated Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return best_model