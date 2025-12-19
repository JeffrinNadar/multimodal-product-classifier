import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import time
from tqdm import tqdm

class BaselineModels:
    """
    Classical ML models for text classification baseline
    """
    def __init__(self, max_features=10000):
        """
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        
        self.models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                n_jobs=-1
            ),
            'linear_svm': LinearSVC(
                max_iter=1000,
                C=1.0,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'naive_bayes': MultinomialNB(alpha=1.0)
        }
        
        self.results = {}
    
    def prepare_data(self, train_df, val_df, test_df, text_column='text'):
        """
        Vectorize text data using TF-IDF
        
        Args:
            train_df, val_df, test_df: Dataframes with text
            text_column: Column containing text
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Vectorizing text with TF-IDF...")
        
        # Fit vectorizer on training data
        X_train = self.vectorizer.fit_transform(train_df[text_column])
        X_val = self.vectorizer.transform(val_df[text_column])
        X_test = self.vectorizer.transform(test_df[text_column])
        
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_all(self, X_train, y_train, X_val, y_val):
        """
        Train all baseline models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        print("\n" + "="*50)
        print("Training Baseline Models")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate on train
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            train_f1 = f1_score(y_train, train_pred, average='weighted')
            
            # Evaluate on validation
            start_time = time.time()
            val_pred = model.predict(X_val)
            inference_time = time.time() - start_time
            
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            # Store results
            self.results[name] = {
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'train_time': train_time,
                'inference_time': inference_time
            }
            
            print(f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Training time: {train_time:.2f}s")
            print(f"Inference time: {inference_time:.4f}s")
    
    def evaluate_test(self, X_test, y_test):
        """
        Evaluate all models on test set
        
        Args:
            X_test, y_test: Test data
        """
        print("\n" + "="*50)
        print("Test Set Evaluation")
        print("="*50)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.results[name]['test_acc'] = test_acc
            self.results[name]['test_f1'] = test_f1
            
            print(f"\n{name}:")
            print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    def get_results_df(self):
        """
        Get results as a formatted dataframe
        
        Returns:
            DataFrame with all results
        """
        results_list = []
        for name, metrics in self.results.items():
            row = {'model': name}
            row.update(metrics)
            results_list.append(row)
        
        df = pd.DataFrame(results_list)
        return df.round(4)
    
    def save_best_model(self, output_path='best_baseline_model.pkl'):
        """
        Save the best performing model
        
        Args:
            output_path: Path to save model
        """
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['val_f1'])[0]
        best_model = self.models[best_model_name]
        
        print(f"\nSaving best model: {best_model_name}")
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'vectorizer': self.vectorizer,
                'model_name': best_model_name
            }, f)
        
        print(f"Model saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Load data
    # train_df = pd.read_csv('data/train.csv')
    # val_df = pd.read_csv('data/val.csv')
    # test_df = pd.read_csv('data/test.csv')
    
    # Initialize baseline models
    baseline = BaselineModels(max_features=10000)
    
    # Prepare data
    # X_train, X_val, X_test, y_train, y_val, y_test = baseline.prepare_data(
    #     train_df, val_df, test_df
    # )
    
    # Train all models
    # baseline.train_all(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    # baseline.evaluate_test(X_test, y_test)
    
    # Print results
    # results_df = baseline.get_results_df()
    # print("\n" + "="*50)
    # print("Final Results Summary")
    # print("="*50)
    # print(results_df.to_string(index=False))
    
    # Save best model
    # baseline.save_best_model()
    
    pass