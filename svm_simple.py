"""
Simple SVM Classification Script
===============================

A comprehensive demonstration of SVM classification with different kernels,
decision boundary visualization, and performance evaluation.

Author: HamsukyTech
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_preprocess_data(file_path):
    """Load and preprocess the churn dataset."""
    print("📁 Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Handle target variable
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'True': 1, 'False': 0, True: 1, False: 0})
    elif df['Churn'].dtype == 'bool':
        df['Churn'] = df['Churn'].astype(int)
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_columns:
        categorical_columns.remove('Churn')
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Features: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

def train_svm_models(X_train, y_train):
    """Train SVM models with different kernels."""
    print("\n⚡ Training SVM models with different kernels...")
    
    kernels = ['linear', 'rbf', 'poly']
    models = {}
    
    for kernel in kernels:
        print(f"Training {kernel} kernel...")
        
        # Define parameter grid based on kernel
        if kernel == 'linear':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif kernel == 'rbf':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}
        else:  # poly
            param_grid = {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}
        
        # Grid search
        svm = SVC(kernel=kernel, probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        models[kernel] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1-score: {grid_search.best_score_:.4f}")
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all SVM models."""
    print("\n📈 Evaluating models...")
    
    results = {}
    
    for kernel, model_info in models.items():
        model = model_info['model']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        results[kernel] = metrics
        
        print(f"\n{kernel.upper()} Kernel Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results

def visualize_performance_comparison(results):
    """Visualize performance comparison between kernels."""
    print("\n📊 Creating performance comparison visualization...")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results).T
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('SVM Kernels Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Kernel', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df

def visualize_roc_curves(models, X_test, y_test):
    """Plot ROC curves for all models."""
    print("\n📈 Creating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for kernel, model_info in models.items():
        model = model_info['model']
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'{kernel.upper()} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.8)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_decision_boundary(models, X_train, y_train, feature_names):
    """Visualize decision boundaries using PCA for dimensionality reduction."""
    print("\n🎯 Creating decision boundary visualizations...")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create subplots for each kernel
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, (kernel, model_info) in enumerate(models.items()):
        ax = axes[idx]
        
        # Train a new SVM on 2D data for visualization
        model_2d = SVC(kernel=kernel, probability=True, random_state=42)
        model_2d.fit(X_train_2d, y_train)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Plot decision boundary
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        scatter = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, 
                           cmap=plt.cm.RdYlBu, edgecolors='black')
        
        # Plot support vectors
        if hasattr(model_2d, 'support_vectors_'):
            ax.scatter(model_2d.support_vectors_[:, 0], model_2d.support_vectors_[:, 1],
                      s=100, facecolors='none', edgecolors='red', linewidth=2)
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title(f'{kernel.upper()} Kernel\nSupport Vectors: {len(model_2d.support_vectors_) if hasattr(model_2d, "support_vectors_") else "N/A"}')
    
    plt.tight_layout()
    plt.show()

def cross_validation_analysis(models, X_train, y_train):
    """Perform cross-validation analysis."""
    print("\n🔄 Performing cross-validation analysis...")
    
    cv_results = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for kernel, model_info in models.items():
        model = model_info['model']
        kernel_results = {}
        
        for metric in metrics:
            scores = cross_val_score(model, X_train, y_train, cv=10, scoring=metric, n_jobs=-1)
            kernel_results[metric] = scores
            print(f"{kernel.upper()} - {metric.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        cv_results[kernel] = kernel_results
    
    # Visualize CV results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        data = []
        labels = []
        for kernel in cv_results.keys():
            data.append(cv_results[kernel][metric])
            labels.append(kernel.upper())
        
        ax.boxplot(data, labels=labels)
        ax.set_title(f'{metric.upper()} Scores')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.show()
    
    return cv_results

def generate_detailed_report(models, results, cv_results):
    """Generate a detailed performance report."""
    print("\n📋 Generating detailed performance report...")
    
    print("="*80)
    print("SVM CLASSIFICATION PERFORMANCE REPORT")
    print("="*80)
    
    # Test set performance
    print("\n📊 TEST SET PERFORMANCE:")
    print("-"*50)
    performance_df = pd.DataFrame(results).T
    print(performance_df.round(4).to_string())
    
    # Cross-validation performance
    print("\n🔄 CROSS-VALIDATION PERFORMANCE (10-fold):")
    print("-"*50)
    cv_summary = []
    for kernel, kernel_results in cv_results.items():
        for metric, scores in kernel_results.items():
            cv_summary.append({
                'Kernel': kernel.upper(),
                'Metric': metric.upper(),
                'Mean': f"{scores.mean():.4f}",
                'Std': f"{scores.std():.4f}"
            })
    
    cv_df = pd.DataFrame(cv_summary)
    print(cv_df.to_string(index=False))
    
    # Best model recommendation
    print("\n🏆 MODEL RECOMMENDATIONS:")
    print("-"*50)
    
    best_f1 = performance_df['F1-Score'].idxmax()
    best_auc = performance_df['AUC'].idxmax()
    best_accuracy = performance_df['Accuracy'].idxmax()
    
    print(f"Best F1-Score: {best_f1.upper()} ({performance_df.loc[best_f1, 'F1-Score']:.4f})")
    print(f"Best AUC: {best_auc.upper()} ({performance_df.loc[best_auc, 'AUC']:.4f})")
    print(f"Best Accuracy: {best_accuracy.upper()} ({performance_df.loc[best_accuracy, 'Accuracy']:.4f})")
    
    # Model characteristics
    print("\n🔍 KERNEL CHARACTERISTICS:")
    print("-"*50)
    print("LINEAR: Fast, interpretable, works well when data is linearly separable")
    print("RBF: Most popular, handles non-linear relationships, good default choice")
    print("POLYNOMIAL: Captures polynomial relationships, can be computationally expensive")
    
    print("\n✅ Analysis completed!")

def main():
    """Main execution function."""
    print("🚀 Starting SVM Classification Analysis")
    print("="*60)
    
    # File path
    file_path = r"c:\Users\ITAPPS002\OneDrive\Documents\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data(file_path)
    
    # Train models
    models = train_svm_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Visualizations
    performance_df = visualize_performance_comparison(results)
    visualize_roc_curves(models, X_test, y_test)
    visualize_decision_boundary(models, X_train, y_train, feature_names)
    
    # Cross-validation analysis
    cv_results = cross_validation_analysis(models, X_train, y_train)
    
    # Generate detailed report
    generate_detailed_report(models, results, cv_results)

if __name__ == "__main__":
    main()
