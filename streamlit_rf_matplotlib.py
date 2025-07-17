"""
Streamlit Random Forest Classifier for Churn Prediction (Matplotlib Version)
============================================================================

Interactive web application for Random Forest classification with hyperparameter tuning,
cross-validation, and feature importance analysis using matplotlib.

Author: hamsukytech
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve, learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
import warnings
import time
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Random Forest Classifier",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRandomForestClassifier:
    """
    Streamlit Random Forest classifier with interactive interface.
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.feature_names = None
        self.label_encoders = {}
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'results' not in st.session_state:
            st.session_state.results = None
            
            #Addedd to save the state of training
        if 'training_in_progress' not in st.session_state:
            st.session_state.training_in_progress = False
        if 'best_model' not in st.session_state:
            st.session_state.best_model = None
        if 'feature_names' not in st.session_state:
            st.session_state.feature_names = None
        if 'X_train' not in st.session_state:
            st.session_state.X_train = None
        if 'X_test' not in st.session_state:
            st.session_state.X_test = None
        if 'y_train' not in st.session_state:
            st.session_state.y_train = None
        if 'y_test' not in st.session_state:
            st.session_state.y_test = None

    def load_data(self, uploaded_file=None, file_path=None):
        """Load data from uploaded file or file path."""
        try:
            if uploaded_file is not None:
                self.df = pd.read_csv(uploaded_file)
            elif file_path is not None:
                self.df = pd.read_csv(file_path)
            else:
                return False
            
            st.session_state.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def display_data_overview(self):
        """Display data overview and statistics."""
        st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", self.df.shape[0])
        with col2:
            st.metric("Total Columns", self.df.shape[1])
        with col3:
            st.metric("Missing Values", self.df.isnull().sum().sum())
        with col4:
            churn_rate = (self.df['Churn'].sum() / len(self.df) * 100) if 'Churn' in self.df.columns else 0
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Display first few rows
        st.subheader("First 5 Rows")
        st.dataframe(self.df.head())
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(self.df.describe())
        
        # Target distribution
        if 'Churn' in self.df.columns:
            st.subheader("Target Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                churn_counts = self.df['Churn'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts.plot(kind='bar', ax=ax)
                ax.set_title("Churn Distribution")
                ax.set_xlabel("Churn")
                ax.set_ylabel("Count")
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title("Churn Distribution (Percentage)")
                ax.set_ylabel("")
                st.pyplot(fig)

    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        with st.spinner("Preprocessing data..."):
            # Make a copy of the dataframe
            df_processed = self.df.copy()
            
            # Convert boolean target to numeric
            if df_processed['Churn'].dtype == 'bool':
                df_processed['Churn'] = df_processed['Churn'].astype(int)
            elif df_processed['Churn'].dtype == 'object':
                # Handle string representations
                churn_map = {'True': 1, 'False': 0, True: 1, False: 0}
                df_processed['Churn'] = df_processed['Churn'].map(churn_map)
            
            # Encode categorical variables
            categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            if 'Churn' in categorical_columns:
                categorical_columns.remove('Churn')
            
            for col in categorical_columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            
            # Separate features and target
            X = df_processed.drop('Churn', axis=1)
            y = df_processed['Churn']
            
            self.feature_names = X.columns.tolist()
            st.session_state.feature_names = self.feature_names
            
            # Split the data
            test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Store in session state
            st.session_state.X_train = self.X_train
            st.session_state.X_test = self.X_test
            st.session_state.y_train = self.y_train
            st.session_state.y_test = self.y_test
            
            st.success(f"✅ Data preprocessed successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Training set: {self.X_train.shape[0]} samples")
            with col2:
                st.info(f"Test set: {self.X_test.shape[0]} samples")

    def train_model(self, use_grid_search=True):
        """Train the Random Forest model with optional hyperparameter tuning."""
        
        # Load data from session state if available
        if (st.session_state.X_train is not None and 
            st.session_state.X_test is not None and 
            st.session_state.y_train is not None and 
            st.session_state.y_test is not None):
            self.X_train = st.session_state.X_train
            self.X_test = st.session_state.X_test
            self.y_train = st.session_state.y_train
            self.y_test = st.session_state.y_test
            self.feature_names = st.session_state.feature_names
        
        if self.X_train is None:
            st.warning("Please preprocess the data first!")
            return
        
        if use_grid_search:
            st.markdown('<h3 class="sub-header">🎯 Hyperparameter Tuning</h3>', unsafe_allow_html=True)
            
            # Parameter grid setup
            st.subheader("Grid Search Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.multiselect(
                    "Number of Estimators", 
                    [50, 100, 200, 300], 
                    default=[100, 200],
                    key="n_estimators"
                )
                max_depth = st.multiselect(
                    "Max Depth", 
                    [10, 20, 30, "None"], 
                    default=[20, "None"],
                    key="max_depth"
                )
            
            with col2:
                min_samples_split = st.multiselect(
                    "Min Samples Split", 
                    [2, 5, 10], 
                    default=[2, 5],
                    key="min_samples_split"
                )
                min_samples_leaf = st.multiselect(
                    "Min Samples Leaf", 
                    [1, 2, 4], 
                    default=[1, 2],
                    key="min_samples_leaf"
                )
            
            max_features = st.multiselect(
                "Max Features", 
                ['sqrt', 'log2'], 
                default=['sqrt'],
                key="max_features"
            )
            
            # Use a form to prevent page refresh
            with st.form("grid_search_form"):
                submitted = st.form_submit_button("🚀 Start Training with Grid Search")
                
                if submitted:
                    if not n_estimators:
                        st.error("Please select at least one value for Number of Estimators")
                        return
                    
                    # Convert "None" string to None for max_depth
                    max_depth_processed = [None if x == "None" else x for x in max_depth]
                    
                    param_grid = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth_processed,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features
                    }
                    
                    total_combinations = (len(n_estimators) * len(max_depth) * 
                                        len(min_samples_split) * len(min_samples_leaf) * 
                                        len(max_features))
                    st.info(f"Grid search will test {total_combinations} parameter combinations")
                    
                    with st.spinner("Training model with Grid Search... This may take a few minutes."):
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        
                        status_text.text("Initializing Grid Search...")
                        progress_bar.progress(10)
                        
                        grid_search = GridSearchCV(
                            estimator=rf,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='f1',
                            n_jobs=-1,
                            verbose=0
                        )
                        
                        status_text.text("Training models...")
                        progress_bar.progress(30)
                        
                        start_time = time.time()
                        grid_search.fit(self.X_train, self.y_train)
                        training_time = time.time() - start_time
                        
                        progress_bar.progress(100)
                        status_text.text("Training completed!")
                        
                        self.best_model = grid_search.best_estimator_
                        st.session_state.best_model = self.best_model
                        st.session_state.model_trained = True
                        
                        # Display results
                        st.success(f"✅ Model trained successfully in {training_time:.2f} seconds!")
                        
                        st.subheader("Best Parameters")
                        best_params_df = pd.DataFrame([grid_search.best_params_]).T
                        best_params_df.columns = ['Value']
                        st.dataframe(best_params_df)
                        
                        st.metric("Best Cross-Validation F1-Score", f"{grid_search.best_score_:.4f}")
                        
                        # Store results for later use
                        st.session_state.grid_search_results = {
                            'best_params': grid_search.best_params_,
                            'best_score': grid_search.best_score_,
                            'training_time': training_time
                        }
        
        else:
            st.markdown('<h3 class="sub-header">🌳 Custom Model Training</h3>', unsafe_allow_html=True)
            
            # Custom parameters
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.slider("Number of Estimators", 50, 500, 100, 50)
                max_depth = st.selectbox("Max Depth", [None, 10, 20, 30, 40])
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            
            with col2:
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
                max_features = st.selectbox("Max Features", ['sqrt', 'log2', None])
                
            with st.form("custom_model_form"):
                submitted = st.form_submit_button("🚀 Train Custom Model")
                
                if submitted:
                    with st.spinner("Training custom model..."):
                        self.best_model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        start_time = time.time()
                        self.best_model.fit(self.X_train, self.y_train)
                        training_time = time.time() - start_time
                        
                        st.session_state.best_model = self.best_model
                        st.session_state.model_trained = True
                        
                        st.success(f"✅ Custom model trained successfully in {training_time:.2f} seconds!")

    def evaluate_model(self):
        """Evaluate the trained model."""
        # Load model from session state if available
        if st.session_state.best_model is not None:
            self.best_model = st.session_state.best_model
        
        if not st.session_state.model_trained or self.best_model is None:
            st.warning("Please train a model first!")
            return
        
        # Load data from session state
        if (st.session_state.X_test is not None and st.session_state.y_test is not None):
            self.X_test = st.session_state.X_test
            self.y_test = st.session_state.y_test
            self.feature_names = st.session_state.feature_names
        
        st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', unsafe_allow_html=True)
        
        # Make predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Display metrics
        st.subheader("Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        cols = [col1, col2, col3, col4, col5]
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(metric, f"{value:.4f}")
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Confusion Matrix", "📈 ROC Curve", "🎯 Precision-Recall", "📋 Classification Report"])
        
        with tab1:
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with tab2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["ROC-AUC"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with tab3:
            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall_curve, precision_curve)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.grid(True)
            st.pyplot(fig)
        
        with tab4:
            # Classification Report
            report = classification_report(self.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        return metrics

    def cross_validation_analysis(self):
        """Perform cross-validation analysis."""
        # Load model from session state if available
        if st.session_state.best_model is not None:
            self.best_model = st.session_state.best_model
        
        if not st.session_state.model_trained or self.best_model is None:
            st.warning("Please train a model first!")
            return
        
        # Load data from session state
        if (st.session_state.X_train is not None and st.session_state.y_train is not None):
            self.X_train = st.session_state.X_train
            self.y_train = st.session_state.y_train
        
        st.markdown('<h2 class="sub-header">🔄 Cross-Validation Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Performing cross-validation..."):
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(
                    self.best_model, self.X_train, self.y_train, 
                    cv=cv, scoring=metric, n_jobs=-1
                )
                cv_results[metric] = scores
            
            # Display CV results
            st.subheader("Cross-Validation Results")
            
            cv_summary = []
            for metric, scores in cv_results.items():
                cv_summary.append({
                    'Metric': metric.upper(),
                    'Mean': f"{scores.mean():.4f}",
                    'Std': f"{scores.std():.4f}",
                    'Min': f"{scores.min():.4f}",
                    'Max': f"{scores.max():.4f}"
                })
            
            cv_df = pd.DataFrame(cv_summary)
            st.dataframe(cv_df, use_container_width=True)
            
            # Visualize CV results
            col1, col2 = st.columns(2)
            
            with col1:
                # Box plot
                cv_scores_df = pd.DataFrame(cv_results)
                fig, ax = plt.subplots(figsize=(10, 6))
                cv_scores_df.boxplot(ax=ax)
                ax.set_title("Cross-Validation Scores Distribution")
                ax.set_ylabel("Score")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                # Learning curve
                train_sizes, train_scores, val_scores = learning_curve(
                    self.best_model, self.X_train, self.y_train, 
                    cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
                ax.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
                ax.set_xlabel('Training Set Size')
                ax.set_ylabel('F1 Score')
                ax.set_title('Learning Curve')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

    def feature_importance_analysis(self):
        """Analyze feature importance."""
        # Load model from session state if available
        if st.session_state.best_model is not None:
            self.best_model = st.session_state.best_model
        
        if not st.session_state.model_trained or self.best_model is None:
            st.warning("Please train a model first!")
            return
        
        # Load feature names from session state
        if st.session_state.feature_names is not None:
            self.feature_names = st.session_state.feature_names
        
        st.markdown('<h2 class="sub-header">🔍 Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        # Get feature importances
        importances = self.best_model.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Display top features
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Most Important Features")
            top_10 = feature_importance_df.head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(top_10['Feature'], top_10['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Feature Importance Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(feature_importance_df['Importance'], bins=20, edgecolor='black')
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Number of Features')
            ax.set_title('Distribution of Feature Importance')
            st.pyplot(fig)
        
        # Cumulative importance
        st.subheader("Cumulative Feature Importance")
        cumulative_importance = np.cumsum(feature_importance_df['Importance'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-')
        ax.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Feature Importance')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Feature importance table
        st.subheader("Complete Feature Importance Ranking")
        st.dataframe(feature_importance_df, use_container_width=True)
        
        # Recommendations
        n_features_80 = np.argmax(cumulative_importance >= 0.8) + 1
        n_features_90 = np.argmax(cumulative_importance >= 0.9) + 1
        
        st.markdown(f"""
        **Feature Selection Recommendations:**
        - To capture 80% of importance: **{n_features_80}** features
        - To capture 90% of importance: **{n_features_90}** features
        - Total features in dataset: **{len(self.feature_names)}** features
        """)
        
        return feature_importance_df

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.markdown('<h1 class="main-header">🌳 Random Forest Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Machine Learning Application for Customer Churn Prediction**")
    
    # Initialize classifier
    classifier = StreamlitRandomForestClassifier()
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Data loading options
    st.sidebar.markdown("### 📁 Data Loading")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Default Churn Dataset"]
    )
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            if classifier.load_data(uploaded_file=uploaded_file):
                st.sidebar.success("✅ Data loaded successfully!")
    else:
        default_path = r"c:\Users\ITAPPS002\OneDrive\Documents\Data Set For Task\Churn Prdiction Data\churn-bigml-80.csv"
        if st.sidebar.button("Load Default Dataset"):
            if classifier.load_data(file_path=default_path):
                st.sidebar.success("✅ Default dataset loaded!")
    
    # Main content
    if st.session_state.data_loaded and classifier.df is not None:
        
        # Data overview
        classifier.display_data_overview()
        
        # Preprocessing
        st.markdown('<h2 class="sub-header">🔧 Data Preprocessing</h2>', unsafe_allow_html=True)
        if st.button("Preprocess Data"):
            classifier.preprocess_data()
        
        # Model training options
        if (classifier.X_train is not None or 
            (st.session_state.X_train is not None and st.session_state.y_train is not None)):
            st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
            
            training_option = st.radio(
                "Choose training method:",
                ["Grid Search (Recommended)", "Custom Parameters"]
            )
            
            if training_option == "Grid Search (Recommended)":
                classifier.train_model(use_grid_search=True)
            else:
                classifier.train_model(use_grid_search=False)
        
        # Model evaluation and analysis
        if st.session_state.model_trained:
            
            # Evaluation tabs
            eval_tab1, eval_tab2, eval_tab3 = st.tabs(["📊 Model Evaluation", "🔄 Cross-Validation", "🔍 Feature Importance"])
            
            with eval_tab1:
                classifier.evaluate_model()
            
            with eval_tab2:
                classifier.cross_validation_analysis()
            
            with eval_tab3:
                classifier.feature_importance_analysis()
            
            # Download model results
            st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)
            
            if st.button("📁 Generate Model Summary Report"):
                # Create summary report
                best_model = st.session_state.best_model if st.session_state.best_model is not None else classifier.best_model
                X_test = st.session_state.X_test if st.session_state.X_test is not None else classifier.X_test
                y_test = st.session_state.y_test if st.session_state.y_test is not None else classifier.y_test
                X_train = st.session_state.X_train if st.session_state.X_train is not None else classifier.X_train
                feature_names = st.session_state.feature_names if st.session_state.feature_names is not None else classifier.feature_names
                
                if best_model is not None and X_test is not None and y_test is not None:
                    y_pred = best_model.predict(X_test)
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    report_data = {
                        'Model Type': 'Random Forest',
                        'Number of Trees': best_model.n_estimators,
                        'Max Depth': best_model.max_depth,
                        'Test Accuracy': accuracy_score(y_test, y_pred),
                        'Test F1-Score': f1_score(y_test, y_pred),
                        'Test ROC-AUC': roc_auc_score(y_test, y_pred_proba),
                        'Training Samples': len(X_train) if X_train is not None else 'N/A',
                        'Test Samples': len(X_test),
                        'Number of Features': len(feature_names) if feature_names is not None else 'N/A'
                    }
                    
                    report_df = pd.DataFrame([report_data]).T
                    report_df.columns = ['Value']
                    
                    st.subheader("📋 Model Summary Report")
                    st.dataframe(report_df)
                    
                    # Convert to CSV for download
                    csv = report_df.to_csv()
                    st.download_button(
                        label="⬇️ Download Report as CSV",
                        data=csv,
                        file_name="random_forest_model_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Model or test data not available. Please train a model first.")
    
    else:
        st.info("👆 Please load a dataset to get started!")
        
        # Show sample data format
        st.markdown("### 📝 Expected Data Format")
        st.markdown("""
        Your CSV file should contain:
        - A target column named 'Churn' (True/False or 1/0)
        - Feature columns with customer data
        - No missing values in critical columns
        
        **Example columns:** State, International plan, Voice mail plan, Number vmail messages, Total day minutes, etc.
        """)

if __name__ == "__main__":
    main()
