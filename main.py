"""
Customer Churn Prediction using Decision Trees
==============================================

This project demonstrates a complete machine learning pipeline for predicting customer churn
using decision trees. The code follows best practices for data science projects and includes
proper data preprocessing, feature engineering, model optimization, and evaluation.

Author: Ali Sekhavati
Date: 27/06/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    f1_score
)
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChurnPredictor:
    """
    A complete churn prediction system using Decision Trees
    
    This class encapsulates the entire machine learning pipeline from data loading
    to model evaluation, making it easy to use and maintain.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
    def load_data(self, file_path):
        """
        Load and perform initial data cleaning
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("Loading and cleaning data...")
        
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Remove non-predictive columns
        df = df.drop('customerID', axis=1)
        
        # Handle TotalCharges column (convert to numeric)
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)  # Fill missing with 0 (new customers)
        
        # Remove any remaining missing values
        df = df.dropna()
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
        
        return df
    
    def explore_data(self, df):
        """
        Perform exploratory data analysis
        
        Args:
            df (pd.DataFrame): Dataset to explore
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        print(f"\nChurn Distribution:")
        for category, count in churn_counts.items():
            print(f"  {category}: {count} ({count/len(df):.2%})")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Churn distribution
        churn_counts.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
        axes[0,0].set_title('Churn Distribution')
        axes[0,0].set_xlabel('Churn')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=0)
        
        # Numerical features vs Churn
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for i, feature in enumerate(numerical_features):
            ax = axes[0,1] if i == 0 else axes[1, i-1]
            sns.boxplot(data=df, x='Churn', y=feature, ax=ax)
            ax.set_title(f'{feature} vs Churn')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis for numerical features
        numerical_df = df.select_dtypes(include=[np.number])
        if len(numerical_df.columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numerical_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            plt.show()
    
    def engineer_features(self, df):
        """
        Create meaningful features for better prediction
        
        Args:
            df (pd.DataFrame): Original dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        print("\nEngineering features...")
        
        df = df.copy()
        
        # 1. Customer value metrics
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)  # Avoid division by zero
        df['IsHighValue'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
        
        # 2. Service engagement
        service_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['TotalServices'] = df[service_features].apply(lambda x: (x == 'Yes').sum(), axis=1)
        df['ServiceEngagement'] = df['TotalServices'] / len(service_features)
        
        # 3. Contract and payment risk
        df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['HasElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
        df['RiskScore'] = df['IsMonthToMonth'] + df['HasElectronicCheck']  # Simple risk score
        
        # 4. Customer lifecycle
        df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
        df['IsLoyalCustomer'] = (df['tenure'] >= 24).astype(int)
        
        # 5. Family status
        df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
        
        print(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def preprocess_data(self, df):
        """
        Prepare data for machine learning
        
        Args:
            df (pd.DataFrame): Dataset with engineered features
            
        Returns:
            tuple: (X, y) features and target
        """
        print("Preprocessing data...")
        
        # Apply feature engineering
        df = self.engineer_features(df)
        
        # Convert categorical variables to dummy variables
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Separate features and target
        X = df_encoded.drop('Churn_Yes', axis=1)
        y = df_encoded['Churn_Yes']
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"Preprocessing completed. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y
    
    def optimize_model(self, X_train, y_train):
        """
        Find the best hyperparameters using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            DecisionTreeClassifier: Optimized model
        """
        print("Optimizing model hyperparameters...")
        
        # Define parameter grid for optimization
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced']
        }
        
        # Create base model
        dt = DecisionTreeClassifier(random_state=self.random_state)
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=dt,
            param_grid=param_grid,
            cv=5,
            scoring='f1',  # F1 score is good for imbalanced datasets
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xlabel('Predicted')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1,0].plot(recall, precision)
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1,1].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1,1].set_yticks(range(len(feature_importance)))
            axes[1,1].set_yticklabels(feature_importance['feature'])
            axes[1,1].set_xlabel('Importance')
            axes[1,1].set_title('Top 10 Feature Importances')
            axes[1,1].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def cross_validate_model(self, model, X, y):
        """
        Perform cross-validation to assess model stability
        
        Args:
            model: Trained model
            X: Features
            y: Target
        """
        print("\n" + "="*30)
        print("CROSS-VALIDATION RESULTS")
        print("="*30)
        
        # Perform 5-fold cross-validation
        cv_f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        cv_roc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        print(f"F1 Scores: {cv_f1_scores}")
        print(f"Mean F1: {cv_f1_scores.mean():.4f} (±{cv_f1_scores.std():.4f})")
        print(f"Mean ROC-AUC: {cv_roc_scores.mean():.4f} (±{cv_roc_scores.std():.4f})")
        
        return cv_f1_scores, cv_roc_scores
    
    def fit(self, file_path):
        """
        Complete training pipeline
        
        Args:
            file_path (str): Path to the training data
        """
        # Load and explore data
        df = self.load_data(file_path)
        self.explore_data(df)
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nData split completed:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        print(f"  Training churn rate: {y_train.mean():.3f}")
        print(f"  Test churn rate: {y_test.mean():.3f}")
        
        # Optimize and train model
        self.model = self.optimize_model(X_train, y_train)
        
        # Evaluate model
        results = self.evaluate_model(self.model, X_test, y_test)
        
        # Cross-validation
        cv_f1, cv_roc = self.cross_validate_model(self.model, X, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return results
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features for prediction
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """
    Main function to run the complete churn prediction pipeline
    """
    # Initialize the churn predictor
    predictor = ChurnPredictor(random_state=42)
    
    # Path to your dataset
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    try:
        # Run the complete pipeline
        results = predictor.fit(file_path)
        
        # Get and save feature importance
        feature_importance = predictor.get_feature_importance()
        feature_importance.to_csv('feature_importance.csv', index=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        print(f"\nFeature importance saved to 'feature_importance.csv'")
        print(f"Model training completed successfully!")
        
        return predictor
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'")
        print("Please make sure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the main pipeline
    trained_predictor = main()
    
    if trained_predictor:
        print("\n" + "="*50)
        print("READY FOR PRODUCTION!")
        print("="*50)
        print("The model is now trained and ready to make predictions on new data.")
        print("Use trained_predictor.predict(new_data) to make predictions.")
