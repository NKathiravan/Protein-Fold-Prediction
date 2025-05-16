import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset (replace with your actual data file)
data = pd.read_csv("D:/KATHIRAVAN/SEM5/Project/datasets/casp12.csv")

# Preprocessing
# Encode amino acids and structural classes
le_aa = LabelEncoder()
data['aa_encoded'] = le_aa.fit_transform(data['aa'])

le_q3 = LabelEncoder()
data['q3_encoded'] = le_q3.fit_transform(data['q3'])  # Use q8_encoded if predicting Q8 classes

# Select features and target
X = data[['aa_encoded', 'asa', 'rsa', 'phi', 'psi']]
y = data['q3_encoded']  # Change to 'q8_encoded' if predicting Q8 classes

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=le_q3.classes_)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Classification Report:\n", report)

# # Logistic Regression
# evaluate_model(LogisticRegression(), "Logistic Regression")

# # Random Forest Classifier
# evaluate_model(RandomForestClassifier(), "Random Forest")

# # SVM (Linear SVC)
# evaluate_model(LinearSVC(), "SVM (Linear)")

# # K-Nearest Neighbors Classifier
# evaluate_model(KNeighborsClassifier(n_neighbors=3), "K-Nearest Neighbors")

# # Naive Bayes
# evaluate_model(GaussianNB(), "Naive Bayes")

# # Decision Tree Classifier
# evaluate_model(DecisionTreeClassifier(), "Decision Tree")

# # XGBoost Classifier
# evaluate_model(XGBClassifier(), "XGBoost")

# Multi-Layer Perceptron (MLP) Classifier
evaluate_model(MLPClassifier(hidden_layer_sizes=(100,4), max_iter=500, random_state=42), "MLP (Neural Network)")
