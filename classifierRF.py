import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from matplotlib import pyplot as plt

import winsound

def RF(featuresGait, demo_walk):
    featuresGait = pd.read_csv('featuresGaittemp.csv')
    demo_walk = pd.read_csv('demo_walktemp.csv')

    # Normalize features
    scaler = MinMaxScaler()
    numeric_columns = featuresGait.select_dtypes(include=['float64', 'int64']).columns
    scaler.fit(featuresGait[numeric_columns])
    # Transform the features to their normalized values
    featuresGait[numeric_columns] = scaler.transform(featuresGait[numeric_columns])
    featuresGait = pd.DataFrame(featuresGait, columns=featuresGait.columns)

    featuresGait = pd.concat([demo_walk, featuresGait], axis=1)
    feats = featuresGait.loc[:, 'MSI':]
    # Extract column names
    feature_names = feats.columns

    # Convert DataFrame to NumPy array
    X = feats.values
    y = featuresGait.loc[:, 'professional-diagnosis']
    y = y.values.ravel()

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the cross-validation splitters
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform nested cross-validation for hyperparameter tuning and model evaluation
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv_inner)

    # Perform outer cross-validation loop
    outer_scores = []
    feature_importances = np.zeros(len(feature_names))

    for train_index, val_index in cv_outer.split(X_train, y_train):
        X_train_outer, X_val = X_train[train_index, :], X_train[val_index, :]
        y_train_outer, y_val = y_train[train_index], y_train[val_index]

        # Perform inner cross-validation loop for hyperparameter tuning
        grid_search.fit(X_train_outer, y_train_outer)
        best_model = grid_search.best_estimator_

        # Evaluate the best model on the validation set
        val_score = best_model.score(X_val, y_val)
        outer_scores.append(val_score)

        # Accumulate feature importances
        feature_importances += best_model.feature_importances_

    # Calculate the average validation score across the outer cross-validation loops
    average_score = np.mean(outer_scores)

    # Fit the best model on the entire training set
    best_model.fit(X_train, y_train)

    # Sound
    duration = 1000  # milliseconds
    frequency = 440  # Hz (middle A)
    winsound.Beep(frequency, duration)

    # Make predictions on the testing set
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Calculate classification accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_prob)

    # Create ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Print the average validation score, testing score, accuracy, and AUC
    print("Average Validation Score:", average_score)
    print("Testing Score:", best_model.score(X_test, y_test))
    print("Accuracy:", accuracy)
    print("AUC:", auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing the random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel

    # Feature importance report
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort the DataFrame by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

    # Select the top 30 features
    top_features = importance_df.tail(30)

    # Plot the feature importance for the top 30 features (upside-down)
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Top 30 Feature Importance (Upside-down)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return accuracy, auc, fpr, tpr, importance_df