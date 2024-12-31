import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import plotly.express as px # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings('ignore')

application_record = pd.read_csv("application_record.csv")
credit_record = pd.read_csv("credit_record.csv")

# Explore data shape, column's datatype and nulls existence
print(application_record.info())

# Check for duplicates
duplicate = application_record.duplicated()
print(duplicate.sum())

# Unique values count 
print(application_record.nunique())


application_record.drop(['FLAG_MOBIL', 'OCCUPATION_TYPE'], axis=1, inplace=True)
application_record['CNT_FAM_MEMBERS'] = application_record['CNT_FAM_MEMBERS'].astype("int64")
application_record['AGE'] = np.abs(application_record['DAYS_BIRTH']) // 365
application_record = application_record.drop(columns=['DAYS_BIRTH'])
application_record['YEARS_EMPLOYED'] = np.abs(application_record['DAYS_EMPLOYED']) // 365
application_record = application_record.drop(columns=['DAYS_EMPLOYED'])

print(application_record.head())


# Explore data shape, column's datatype and nulls existence
print(credit_record.info())

# Check for duplicates
duplicate = credit_record.duplicated()
print(duplicate.sum())

# Unique values count 
print(credit_record.nunique())


# 1) Duration
credit_record['DURATION'] = credit_record.groupby('ID')['MONTHS_BALANCE'].transform(lambda x: (x.min() * -1) + 1)


# 2) Trend
status_map = {'X': 0, 'C': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6}
credit_record['STATUS_NUMERIC'] = credit_record['STATUS'].map(status_map)
credit_record['MONTHS_BALANCE'] = credit_record.groupby('ID')['MONTHS_BALANCE'].transform(
    lambda x: x - x.min()
)
credit_record.sort_values(['ID', 'MONTHS_BALANCE'], inplace=True)

# Calculate the overall difference between each customer's status to see the change in it over months 
# (better, worse, no change)
credit_record['CHANGE'] = credit_record.groupby('ID')['STATUS_NUMERIC'].diff(-1)

def determine_trend(changes):
    ups = (changes > 0).sum()
    downs = (changes < 0).sum()
    if ups > downs:
        return 'Better'
    elif downs > ups:
        return 'Worse'
    else:
        return 'No Change'

trend = credit_record.groupby('ID')['CHANGE'].apply(determine_trend)
credit_record = credit_record.merge(trend.rename('TREND'), on='ID', how='left')



# Include only needed features from credit_recort
# Grouping customers to have one record representing each to prevent redundancy
# Merge with inner join
credit_to_merge = credit_record.groupby('ID').agg({
    'DURATION': 'first',
    'TREND': 'first'
}).reset_index()
df = application_record.merge(credit_to_merge, on='ID', how='inner')


# Define target column based on previously mentioned criteria
def credit_labeling(status):
    return 'Good_Credit' if status.isin(['C', 'X']).all() else 'Bad_Credit'

label = credit_record.groupby('ID').agg({
    'STATUS': credit_labeling,
}).reset_index()
df = df.merge(label, on='ID', how='left')

print(df.head())
print(df.info())



print(df['STATUS'].value_counts())

sns.countplot(data=df, x='STATUS', palette='viridis')
plt.title('Distribution of Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()



numerical_col = ['AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED', 'DURATION']
plt.figure(figsize=(15, len(numerical_col) * 4))
for i, col in enumerate(numerical_col, 1):
    plt.subplot(len(numerical_col), 1, i)
    sns.histplot(data=df, x=col, hue='STATUS', kde=True, color='blue', bins=30, multiple='dodge')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    
plt.show()


df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].clip(upper=60) 

sns.histplot(data=df, x='YEARS_EMPLOYED', kde=True, color='blue', bins=30)
plt.title(f'Distribution of YEARS_EMPLOYED')




categorical_col = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
                   'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'TREND']

plt.figure(figsize=(10, len(categorical_col) * 4))
for i, col in enumerate(categorical_col):
    plt.subplot(len(categorical_col), 1, i + 1)
    sns.countplot(data=df, x=col, hue='STATUS', palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()




# Applying Chi-square test on categorical attributes
for attribute in categorical_col:
    contingency_table = pd.crosstab(df['STATUS'], df[attribute])
    p = chi2_contingency(contingency_table)[1]
    if p < 0.05:
        print(f"Attribute: {attribute}")


categorical_col = categorical_col + ['STATUS']
label_encoder = LabelEncoder()
for col in categorical_col:
    df[col] = label_encoder.fit_transform(df[col])
print(df.head())



scaler = StandardScaler()
df['AMT_INCOME_TOTAL'] = scaler.fit_transform(df[['AMT_INCOME_TOTAL']])
print(df.head())



# Calculate Correlation between all attributes and STATUS
correlation_matrix = df.corr()
status_correlation = correlation_matrix['STATUS'].sort_values(ascending=False)
print(status_correlation)

plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Correlation of STATUS with Numerical Features')
plt.show()



X = df.drop(columns=['STATUS', 'ID'])
y = df['STATUS']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)




model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on the training set
y_train_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))


# Generate learning curve data
train_sizes, train_scores, validation_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(validation_scores, axis=1)
valid_scores_std = np.std(validation_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Accuracy")
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)

plt.plot(train_sizes, valid_scores_mean, label="Validation Accuracy")
plt.fill_between(
    train_sizes,
    valid_scores_mean - valid_scores_std,
    valid_scores_mean + valid_scores_std,
    alpha=0.2
)
plt.title("Logisic Regression Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


tree_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=10)
tree_model.fit(X_train, y_train)

# Evaluate on training set
y_train_pred = tree_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")
    
# Evaluate on validation set
y_val_pred = tree_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Evaluate on test set
y_test_pred = tree_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")


# Generate learning curve data
train_sizes, train_scores, validation_scores = learning_curve(
    tree_model, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(validation_scores, axis=1)
valid_scores_std = np.std(validation_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Accuracy")
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)

plt.plot(train_sizes, valid_scores_mean, label="Validation Accuracy")
plt.fill_between(
    train_sizes,
    valid_scores_mean - valid_scores_std,
    valid_scores_mean + valid_scores_std,
    alpha=0.2
)
plt.title("Decision Tree Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)

# Evaluate on training set
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

# Evaluate on validation set
y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# Evaluate on test set
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")

# Generate learning curve data
train_sizes, train_scores, validation_scores = learning_curve(
    rf_model, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(validation_scores, axis=1)
valid_scores_std = np.std(validation_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Accuracy")
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)

plt.plot(train_sizes, valid_scores_mean, label="Validation Accuracy")
plt.fill_between(
    train_sizes,
    valid_scores_mean - valid_scores_std,
    valid_scores_mean + valid_scores_std,
    alpha=0.2
)
plt.title("Random Forest Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 8, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Parameters: {test_accuracy}")





# Generate learning curve data
train_sizes, train_scores, validation_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(validation_scores, axis=1)
valid_scores_std = np.std(validation_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Accuracy")
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2
)

plt.plot(train_sizes, valid_scores_mean, label="Validation Accuracy")
plt.fill_between(
    train_sizes,
    valid_scores_mean - valid_scores_std,
    valid_scores_mean + valid_scores_std,
    alpha=0.2
)
plt.title("Random Forest Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



print("Classification Report:")
print(classification_report(y_test, y_pred))


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))