import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

# Function to load and clean the dataset
def load_and_clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop missing values and 'customerID'
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert 'TotalCharges' to numeric
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    return df

# Function for data exploration (descriptive statistics and visualizations)
def explore_data(df):
    # Show the first few rows of the dataset
    print(df.head())
    
    # Get basic info about the dataset
    print("\nDataset Info:")
    df.info()

    # Display summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    print("\nColumns with Missing Values:")
    print(missing_values)

    # Explore categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("\nCategorical Columns & Unique Values:")
    for col in categorical_cols:
        print(f"{col}: {df[col].unique()}")

    # Visualizations
    sns.countplot(x='Churn', data=df)
    sns.boxplot(x='Churn', y='tenure', data=df)
    plt.show()

# Function to preprocess data (handle categorical variables)
def preprocess_data(df):
    # Convert categorical columns to numerical using OneHotEncoder
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Fill missing values with the median for numerical columns
    df_encoded.fillna(df_encoded.median(), inplace=True)
    
    # Split data into features (X) and target (y)
    X = df_encoded.drop(columns=['Churn_Yes'])
    y = df_encoded['Churn_Yes']
    
    return X, y

# Function to train and evaluate Decision Tree model
def train_decision_tree(X_train, X_test, y_train, y_test):
    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    print("\nDecision Tree Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Plot the decision tree (Optional, only for small trees)
    plt.figure(figsize=(15,10))
    plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['No Churn', 'Churn'])
    plt.show()

# Main function to execute the workflow
def main(file_path):
    # Load and clean data
    df = load_and_clean_data(file_path)
    
    # Explore data
    explore_data(df)
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Decision Tree model
    train_decision_tree(X_train, X_test, y_train, y_test)

# Run the main function with the path to the dataset
if __name__ == "__main__":
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Modify path if needed
    main(file_path)
