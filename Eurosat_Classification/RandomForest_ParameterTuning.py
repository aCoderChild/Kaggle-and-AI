import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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

clf = RandomForestClassifier(max_features=None, n_estimators=100,
                             random_state=12, criterion='entropy')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))
