"""
Random Forest Classifier for Churn Prediction
===============================================

This script implements a comprehensive Random Forest model for customer churn prediction.
It includes hyperparameter tuning, cross-validation, and feature importance analysis.

Author: HamsukyTech
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChurnRandomForestClassifier:
    """
    A comprehensive Random Forest classifier for customer churn prediction.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.feature_names = None
        self.label_encoders = {}
        
    def load_and_explore_data(self):
        """Load and perform initial exploration of the dataset."""
        print("🔍 Loading and Exploring the Dataset")
        print("=" * 50)
        
        # Load the dataset
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nDataset info:")
        print(self.df.info())
        
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        print(f"\nTarget variable distribution:")
        print(self.df['Churn'].value_counts())
        
        print(f"\nMissing values:")
        print(self.df.isnull().sum().sum())
        
        # Visualize target distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        self.df['Churn'].value_counts().plot(kind='bar')
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        self.df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Churn Distribution (Percentage)')
        
        plt.tight_layout()
        plt.show()
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("\n🔧 Preprocessing the Data")
        print("=" * 50)
        
        # Make a copy of the dataframe
        df_processed = self.df.copy()
        
        # Convert boolean target to numeric
        df_processed['Churn'] = df_processed['Churn'].map({True: 1, False: 0})
        
        # Encode categorical variables
        categorical_columns = ['State', 'International plan', 'Voice mail plan']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
        
        # Separate features and target
        X = df_processed.drop('Churn', axis=1)
        y = df_processed['Churn']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def baseline_model(self):
        """Train a baseline Random Forest model."""
        print("\n🌳 Training Baseline Random Forest Model")
        print("=" * 50)
        
        # Create baseline model
        baseline_rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        baseline_rf.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = baseline_rf.predict(self.X_test)
        y_pred_proba = baseline_rf.predict_proba(self.X_test)[:, 1]
        
        # Evaluate baseline model
        print("Baseline Model Performance:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        return baseline_rf
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using GridSearchCV."""
        print("\n🎯 Hyperparameter Tuning")
        print("=" * 50)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform grid search
        print("Performing Grid Search (this may take a few minutes)...")
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Store best model
        self.best_model = grid_search.best_estimator_
        
        print(f"\nBest Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest Cross-Validation F1-Score: {grid_search.best_score_:.4f}")
        
        return self.best_model
    
    def cross_validation_evaluation(self):
        """Perform comprehensive cross-validation evaluation."""
        print("\n📊 Cross-Validation Evaluation")
        print("=" * 50)
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Define scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.best_model, self.X_train, self.y_train, 
                cv=cv, scoring=metric, n_jobs=-1
            )
            cv_results[metric] = scores
            
            print(f"{metric.upper()}:")
            print(f"  Mean: {scores.mean():.4f}")
            print(f"  Std:  {scores.std():.4f}")
            print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
            print()
        
        # Visualize cross-validation results
        plt.figure(figsize=(12, 8))
        
        # Box plot of CV scores
        plt.subplot(2, 2, 1)
        cv_df = pd.DataFrame(cv_results)
        cv_df.boxplot()
        plt.title('Cross-Validation Scores Distribution')
        plt.xticks(rotation=45)
        
        # Learning curve
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, self.X_train, self.y_train, 
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.subplot(2, 2, 2)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        # Validation curve for n_estimators
        param_range = [50, 100, 150, 200, 250, 300]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42), self.X_train, self.y_train,
            param_name='n_estimators', param_range=param_range,
            cv=5, scoring='f1', n_jobs=-1
        )
        
        plt.subplot(2, 2, 3)
        plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(param_range, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
        plt.xlabel('Number of Trees')
        plt.ylabel('F1 Score')
        plt.title('Validation Curve (n_estimators)')
        plt.legend()
        plt.grid(True)
        
        # Validation curve for max_depth
        param_range = [5, 10, 15, 20, 25, 30]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42), self.X_train, self.y_train,
            param_name='max_depth', param_range=param_range,
            cv=5, scoring='f1', n_jobs=-1
        )
        
        plt.subplot(2, 2, 4)
        plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training Score')
        plt.plot(param_range, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
        plt.xlabel('Max Depth')
        plt.ylabel('F1 Score')
        plt.title('Validation Curve (max_depth)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return cv_results
    
    def final_evaluation(self):
        """Evaluate the final model on the test set."""
        print("\n🎯 Final Model Evaluation on Test Set")
        print("=" * 50)
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print("Final Model Performance:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # ROC Curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 3)
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Feature Importance (Top 10)
        plt.subplot(2, 3, 4)
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        plt.bar(range(10), importances[indices])
        plt.title('Top 10 Feature Importances')
        plt.xticks(range(10), [self.feature_names[i] for i in indices], rotation=45)
        
        # Prediction Probability Distribution
        plt.subplot(2, 3, 5)
        plt.hist(y_pred_proba[self.y_test == 0], alpha=0.5, label='No Churn', bins=30)
        plt.hist(y_pred_proba[self.y_test == 1], alpha=0.5, label='Churn', bins=30)
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # Model Performance Metrics Bar Chart
        plt.subplot(2, 3, 6)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [accuracy, precision, recall, f1, roc_auc]
        
        bars = plt.bar(metrics, values)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def feature_importance_analysis(self):
        """Perform detailed feature importance analysis."""
        print("\n🔍 Feature Importance Analysis")
        print("=" * 50)
        
        # Get feature importances
        importances = self.best_model.feature_importances_
        
        # Create DataFrame for better handling
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance Ranking:")
        print(feature_importance_df.to_string(index=False))
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # All features importance
        plt.subplot(2, 2, 1)
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.title('All Features Importance')
        plt.xlabel('Importance')
        
        # Top 10 features
        plt.subplot(2, 2, 2)
        top_10 = feature_importance_df.head(10)
        plt.barh(top_10['feature'], top_10['importance'])
        plt.title('Top 10 Most Important Features')
        plt.xlabel('Importance')
        
        # Cumulative importance
        plt.subplot(2, 2, 3)
        cumulative_importance = np.cumsum(feature_importance_df['importance'])
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        plt.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.legend()
        plt.grid(True)
        
        # Feature importance distribution
        plt.subplot(2, 2, 4)
        plt.hist(importances, bins=20, edgecolor='black')
        plt.xlabel('Feature Importance')
        plt.ylabel('Number of Features')
        plt.title('Feature Importance Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Feature selection recommendations
        cumulative_importance = np.cumsum(feature_importance_df['importance'])
        n_features_80 = np.argmax(cumulative_importance >= 0.8) + 1
        n_features_90 = np.argmax(cumulative_importance >= 0.9) + 1
        
        print(f"\nFeature Selection Recommendations:")
        print(f"• To capture 80% of importance: {n_features_80} features")
        print(f"• To capture 90% of importance: {n_features_90} features")
        print(f"• Total features in dataset: {len(self.feature_names)}")
        
        # Most important features for interpretation
        print(f"\nTop 5 Most Important Features for Churn Prediction:")
        for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows(), 1):
            print(f"{i}. {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance_df
    
    def run_complete_analysis(self):
        """Run the complete Random Forest analysis pipeline."""
        print("🚀 Starting Complete Random Forest Analysis for Churn Prediction")
        print("=" * 70)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Preprocess data
        self.preprocess_data()
        
        # Step 3: Baseline model
        baseline_model = self.baseline_model()
        
        # Step 4: Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Step 5: Cross-validation evaluation
        cv_results = self.cross_validation_evaluation()
        
        # Step 6: Final evaluation
        final_metrics = self.final_evaluation()
        
        # Step 7: Feature importance analysis
        feature_importance_df = self.feature_importance_analysis()
        
        print("\n🎉 Analysis Complete!")
        print("=" * 70)
        print("Summary of Results:")
        print(f"• Best Model F1-Score: {final_metrics['f1_score']:.4f}")
        print(f"• Best Model ROC-AUC: {final_metrics['roc_auc']:.4f}")
        print(f"• Most Important Feature: {feature_importance_df.iloc[0]['feature']}")
        print(f"• Model Type: Random Forest with {self.best_model.n_estimators} trees")
        
        return {
            'model': self.best_model,
            'metrics': final_metrics,
            'feature_importance': feature_importance_df,
            'cv_results': cv_results
        }

# Main execution
if __name__ == "__main__":
    # Initialize the classifier
    data_path = r"c:\Users\ITAPPS002\OneDrive\Documents\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
    
    classifier = ChurnRandomForestClassifier(data_path)
    
    # Run complete analysis
    results = classifier.run_complete_analysis()
    
    print("\n📝 Analysis completed successfully!")
    print("All visualizations and metrics have been generated.")
