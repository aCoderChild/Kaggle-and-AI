import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Load dataset
data_path = 'EuroSAT_13bands.csv'
df = pd.read_csv(data_path)

# Remove spaces from column names if needed
df.columns = df.columns.str.strip()

X_labels = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8',
            'Band9', 'Band10', 'Band11', 'Band12', 'Band13']
y_labels = 'Label'

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[X_labels], df[y_labels], test_size=0.2, stratify=df[y_labels], random_state=42
)

# Initialize classifier
clf = RandomForestClassifier(random_state=9)

# Define cross-validation strategy
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=12)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 100],
    'max_features': [None, 'sqrt'],  # Replacing 'auto' with None
    'criterion': ['gini', 'entropy']
}

# Perform grid search with cross-validation
gridCV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=skf, scoring='accuracy')
gridCV_clf.fit(X_train, y_train)

# Display the best parameters and corresponding score
print(gridCV_clf.best_params_)
print(gridCV_clf.best_score_)
