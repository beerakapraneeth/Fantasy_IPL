import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load training data
train_captain = pd.read_csv("train_data_Captain.csv")
train_total_fp = pd.read_csv("train_data_Total_FP.csv")

# Merge the two training datasets
train_data = pd.merge(train_captain, train_total_fp, on=['match_id', 'season', 'match_name', 'home_team', 'away_team', 'venue', 'batting_innings', 'bowling_innings', 'Player_name', 'Starting_11'])

# Adding new features based on domain knowledge
train_data['total_prev_runs'] = train_data['prev_runs_x'] + train_data['prev_runs_y']
train_data['total_prev_wickets'] = train_data['prev_wickets_x'] + train_data['prev_wickets_y']
train_data['run_rate_x'] = train_data['prev_runs_x'] / (train_data['prev_balls_x'] + 1)
train_data['run_rate_y'] = train_data['prev_runs_y'] / (train_data['prev_balls_y'] + 1)
train_data['total_run_rate'] = train_data['run_rate_x'] + train_data['run_rate_y']

# Check if 'prev_innings_x' and 'prev_innings_y' exist and create averages accordingly
if 'prev_innings_x' in train_data.columns and 'prev_innings_y' in train_data.columns:
    train_data['batting_avg_x'] = train_data['prev_runs_x'] / (train_data['prev_innings_x'] + 1)
    train_data['bowling_avg_x'] = train_data['prev_conceded_x'] / (train_data['prev_wickets_x'] + 1)
    train_data['batting_avg_y'] = train_data['prev_runs_y'] / (train_data['prev_innings_y'] + 1)
    train_data['bowling_avg_y'] = train_data['prev_conceded_y'] / (train_data['prev_wickets_y'] + 1)
else:
    train_data['batting_avg_x'] = train_data['prev_runs_x'] / (train_data['batting_innings'] + 1)
    train_data['bowling_avg_x'] = train_data['prev_conceded_x'] / (train_data['prev_wickets_x'] + 1)
    train_data['batting_avg_y'] = train_data['prev_runs_y'] / (train_data['batting_innings'] + 1)
    train_data['bowling_avg_y'] = train_data['prev_conceded_y'] / (train_data['prev_wickets_y'] + 1)

# Feature Selection (including new features)
features = ['prev_runs_x', 'prev_balls_x', 'prev_sixes_x', 'prev_fours_x', 'prev_wickets_x', 'prev_conceded_x',
            'prev_catches_x', 'prev_Dream Team_x', 'prev_Total_FP_x', 'prev_overs_x', 'prev_fielding_heroics_x',
            'prev_duck_x', 'luck_x', 'prev_runs_y', 'prev_balls_y', 'prev_sixes_y', 'prev_fours_y', 'prev_wickets_y',
            'prev_conceded_y', 'prev_catches_y', 'prev_Dream Team_y', 'prev_Total_FP_y', 'prev_overs_y',
            'prev_fielding_heroics_y', 'prev_duck_y', 'luck_y', 'total_prev_runs', 'total_prev_wickets',
            'run_rate_x', 'run_rate_y', 'total_run_rate', 'batting_avg_x', 'bowling_avg_x', 'batting_avg_y', 'bowling_avg_y']

X = train_data[features]
y_captain = train_data['Captain/Vice Captain']
y_total_fp = train_data['Total_FP']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels for captaincy/vice captaincy prediction
label_encoder = LabelEncoder()
y_captain_encoded = label_encoder.fit_transform(y_captain)

# Train-test split
X_train, X_val, y_captain_train, y_captain_val, y_total_fp_train, y_total_fp_val = train_test_split(
    X_scaled, y_captain_encoded, y_total_fp, test_size=0.2, random_state=42)

# Define base models for stacking
xgb_model_total_fp = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
rf_model_total_fp = RandomForestRegressor(random_state=42)

xgb_model_captain = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
rf_model_captain = RandomForestClassifier(random_state=42)

# Parameter grid for RandomizedSearchCV
param_grid_total_fp = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5]
}

param_grid_captain = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5]
}

# Randomized search for hyperparameter tuning
random_search_total_fp = RandomizedSearchCV(xgb_model_total_fp, param_distributions=param_grid_total_fp, n_iter=100, cv=3, n_jobs=-1, random_state=42)
random_search_captain = RandomizedSearchCV(xgb_model_captain, param_distributions=param_grid_captain, n_iter=100, cv=3, n_jobs=-1, random_state=42)

# Fit the models
random_search_total_fp.fit(X_train, y_total_fp_train)
random_search_captain.fit(X_train, y_captain_train)

# Feature selection using model feature importances
selector = SelectFromModel(random_search_total_fp.best_estimator_, prefit=True, max_features=20)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

# Stacking models
stacking_regressor = StackingRegressor(
    estimators=[('xgb', random_search_total_fp.best_estimator_), ('rf', rf_model_total_fp)],
    final_estimator=LinearRegression()
)

stacking_classifier = StackingClassifier(
    estimators=[('xgb', random_search_captain.best_estimator_), ('rf', rf_model_captain)],
    final_estimator=LogisticRegression()
)

# Define pipelines for polynomial features, imputation, and scaling
pipeline_total_fp = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', stacking_regressor)
])

pipeline_captain = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', stacking_classifier)
])

# Fit pipelines on training data
pipeline_total_fp.fit(X_train_selected, y_total_fp_train)
pipeline_captain.fit(X_train_selected, y_captain_train)

# Make predictions
total_fp_predictions = pipeline_total_fp.predict(X_val_selected)
captain_predictions = pipeline_captain.predict(X_val_selected)

# Decode captaincy/vice captaincy predictions
captain_predictions_decoded = label_encoder.inverse_transform(captain_predictions)

# Calculate accuracy and RMSE
accuracy = accuracy_score(y_captain_val, captain_predictions)
rmse = np.sqrt(mean_squared_error(y_total_fp_val, total_fp_predictions))

print("Accuracy:", accuracy)
print("RMSE:", rmse)

# Load test data
test_captain = pd.read_csv("test_data_Captain.csv")
test_total_fp = pd.read_csv("test_data_Total_FP.csv")

# Merge test datasets
test_data = pd.merge(test_captain, test_total_fp, on=['match_id', 'season', 'match_name', 'home_team', 'away_team', 'venue', 'batting_innings', 'bowling_innings', 'Player_name', 'Starting_11'])

# Feature Engineering on test data
test_data['total_prev_runs'] = test_data['prev_runs_x'] + test_data['prev_runs_y']
test_data['total_prev_wickets'] = test_data['prev_wickets_x'] + test_data['prev_wickets_y']
test_data['run_rate_x'] = test_data['prev_runs_x'] / (test_data['prev_balls_x'] + 1)
test_data['run_rate_y'] = test_data['prev_runs_y'] / (test_data['prev_balls_y'] + 1)
test_data['total_run_rate'] = test_data['run_rate_x'] + test_data['run_rate_y']

# Check if 'prev_innings_x' and 'prev_innings_y' exist and create averages accordingly
if 'prev_innings_x' in test_data.columns and 'prev_innings_y' in test_data.columns:
    test_data['batting_avg_x'] = test_data['prev_runs_x'] / (test_data['prev_innings_x'] + 1)
    test_data['bowling_avg_x'] = test_data['prev_conceded_x'] / (test_data['prev_wickets_x'] + 1)
    test_data['batting_avg_y'] = test_data['prev_runs_y'] / (test_data['prev_innings_y'] + 1)
    test_data['bowling_avg_y'] = test_data['prev_conceded_y'] / (test_data['prev_wickets_y'] + 1)
else:
    test_data['batting_avg_x'] = test_data['prev_runs_x'] / (test_data['batting_innings'] + 1)
    test_data['bowling_avg_x'] = test_data['prev_conceded_x'] / (test_data['prev_wickets_x'] + 1)
    test_data['batting_avg_y'] = test_data['prev_runs_y'] / (test_data['batting_innings'] + 1)
    test_data['bowling_avg_y'] = test_data['prev_conceded_y'] / (test_data['prev_wickets_y'] + 1)

# Ensure test data has the same features as training data
for feature in features:
    if feature not in test_data:
        test_data[feature] = 0

# Ensure the order of features matches the training data
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)

# Ensure the selector uses the correct features
X_test_selected = X_test_scaled[:, selector.get_support()]

# Make predictions on test set using pipelines
total_fp_predictions_test = pipeline_total_fp.predict(X_test_selected)
captain_predictions_test = pipeline_captain.predict(X_test_selected)

# Decode captaincy/vice captaincy predictions
captain_predictions_decoded_test = label_encoder.inverse_transform(captain_predictions_test)

# Format predictions into DataFrame
result_df = pd.DataFrame({
    'match_id': test_data['match_id'],
    'Total_FP': total_fp_predictions_test.round(),
    'Captain/Vice Captain': captain_predictions_decoded_test
})

# Save results to CSV
result_df.to_csv('result.csv', index=False)