import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class SpamClassifier:
    """
    Comprehensive spam classification system with multiple algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.performance_metrics = {}
        
    def initialize_models(self):
        """
        Initialize different classification models
        """
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'naive_bayes_multinomial': MultinomialNB(alpha=1.0),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='linear',
                random_state=42,
                class_weight='balanced',
                probability=True
            )
        }
        
    def handle_imbalanced_data(self, X_train, y_train, method='smote'):
        """
        Handle imbalanced dataset using various techniques
        """
        print(f"Original dataset distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        else:
            return X_train, y_train
        
        print(f"Resampled dataset distribution:")
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(dict(zip(unique, counts)))
        
        return X_resampled, y_resampled
    
    def train_models(self, X_train, y_train, X_test, y_test, use_resampling=True):
        """
        Train all models and evaluate performance
        """
        self.initialize_models()
        
        # Handle imbalanced data if requested
        if use_resampling:
            X_train_resampled, y_train_resampled = self.handle_imbalanced_data(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        print("\nTraining models...")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_resampled, y_train_resampled)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            self.performance_metrics[name] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive performance metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """
        Perform hyperparameter tuning for specific model
        """
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'naive_bayes_multinomial': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 10.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        print(f"\nTuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def select_best_model(self):
        """
        Select the best performing model based on F1-score
        """
        best_f1 = 0
        best_model_name = None
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*50)
        
        for name, metrics in self.performance_metrics.items():
            print(f"\n{name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name.upper()}")
        print(f"F1-Score: {best_f1:.4f}")
        print(f"{'='*50}")
        
        return self.best_model, best_model_name
    
    def save_model(self, model_path='spam_classifier_model.pkl', 
                   preprocessor_path='preprocessor.pkl'):
        """
        Save the best model and preprocessor
        """
        if self.best_model is None:
            print("No model to save. Please train models first.")
            return
        
        # Save model
        joblib.dump(self.best_model, model_path)
        
        # Save preprocessor
        if self.preprocessor is not None:
            joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save performance metrics
        metrics_path = 'model_metrics.pkl'
        joblib.dump(self.performance_metrics, metrics_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")
        print(f"Metrics saved to: {metrics_path}")
        
        # Save model info
        model_info = {
            'best_model_name': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance_metrics': self.performance_metrics[self.best_model_name]
        }
        
        joblib.dump(model_info, 'model_info.pkl')
        print("Model info saved to: model_info.pkl")
    
    def load_model(self, model_path='spam_classifier_model.pkl', 
                   preprocessor_path='preprocessor.pkl'):
        """
        Load saved model and preprocessor
        """
        try:
            self.best_model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Load additional info if available
            try:
                model_info = joblib.load('model_info.pkl')
                self.best_model_name = model_info['best_model_name']
                print(f"Loaded model: {self.best_model_name}")
                print(f"Training date: {model_info['training_date']}")
            except:
                pass
            
            print("Model and preprocessor loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """
        Predict if a text is spam or ham
        """
        if self.best_model is None or self.preprocessor is None:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        # Preprocess text
        vectorized_text, features = self.preprocessor.preprocess_single_text(text)
        
        # Make prediction
        prediction = self.best_model.predict(vectorized_text)[0]
        probability = self.best_model.predict_proba(vectorized_text)[0] if hasattr(self.best_model, 'predict_proba') else None
        
        # Convert prediction to label
        label = self.preprocessor.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'prediction': label,
            'confidence': probability[prediction] if probability is not None else None,
            'raw_prediction': prediction
        }
        
        return result
    
    def plot_confusion_matrix(self, model_name=None):
        """
        Plot confusion matrix for specified model or best model
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.performance_metrics:
            print(f"No metrics available for {model_name}")
            return
        
        cm = self.performance_metrics[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], 
                   yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

def train_spam_classifier(data_file='email_dataset.csv'):
    """
    Complete training pipeline
    """
    from preprocessor import load_and_preprocess_data
    
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, processed_df = load_and_preprocess_data(data_file)
    
    # Initialize classifier
    classifier = SpamClassifier()
    classifier.preprocessor = preprocessor
    
    # Train models
    classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model, best_model_name = classifier.select_best_model()
    
    # Save model
    classifier.save_model()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix()
    
    return classifier

if __name__ == "__main__":
    # Create sample data if needed
    from data_scraper import EmailDataScraper
    
    scraper = EmailDataScraper()
    scraper.scrape_sample_emails()
    df = scraper.create_dataset()
    
    # Train classifier
    classifier = train_spam_classifier()
    
    # Test prediction
    test_spam = "URGENT! You've won $1000000! Click here NOW to claim your prize!"
    test_ham = "Hi John, the meeting is scheduled for tomorrow at 3 PM."
    
    print(f"\nTest predictions:")
    print(f"Text: '{test_spam}'")
    print(f"Prediction: {classifier.predict(test_spam)}")
    
    print(f"\nText: '{test_ham}'")
    print(f"Prediction: {classifier.predict(test_ham)}")