"""
AI-Powered Transaction Fraud Detection System
Complete pipeline for fraud detection using ensemble ML methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_RATIO = 0.5

class FraudDetectionPipeline:
    """Complete fraud detection pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = None
        
    def engineer_features(self, df):
        """Create predictive features from raw transaction data"""
        print("Performing feature engineering...")
        
        df_feat = df.copy()
        
        # Amount features
        df_feat['amount_log'] = np.log1p(df_feat['amount'])
        df_feat['amount_sqrt'] = np.sqrt(df_feat['amount'])
        
        # Balance features
        df_feat['orig_balance_ratio'] = df_feat['amount'] / (df_feat['oldbalanceOrg'] + 1)
        df_feat['dest_balance_ratio'] = df_feat['amount'] / (df_feat['oldbalanceDest'] + 1)
        df_feat['is_orig_zero'] = (df_feat['oldbalanceOrg'] == 0).astype(int)
        df_feat['is_dest_zero'] = (df_feat['oldbalanceDest'] == 0).astype(int)
        df_feat['is_orig_drained'] = (df_feat['newbalanceOrig'] == 0).astype(int)
        
        # Time features
        df_feat['hour'] = df_feat['step'] % 24
        df_feat['day'] = df_feat['step'] // 24
        
        # Transaction type encoding
        df_feat['type_encoded'] = LabelEncoder().fit_transform(df_feat['type'])
        type_dummies = pd.get_dummies(df_feat['type'], prefix='type')
        df_feat = pd.concat([df_feat, type_dummies], axis=1)
        
        # Merchant indicators
        df_feat['is_dest_merchant'] = df_feat['nameDest'].str.startswith('M').astype(int)
        df_feat['is_orig_merchant'] = df_feat['nameOrig'].str.startswith('M').astype(int)
        
        print(f"Created {df_feat.shape[1] - df.shape[1]} new features")
        return df_feat
    
    def preprocess_data(self, df):
        """Preprocess and split data"""
        print("Preprocessing data...")
        
        # Drop unnecessary columns
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'type']
        df_processed = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Separate features and target
        X = df_processed.drop('isFraud', axis=1)
        y = df_processed['isFraud']
        
        # Handle infinite/missing values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        print("\nTraining models...")
        
        # Define models
        self.models = {
            'XGBoost': XGBClassifier(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=0
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            print(f"  ✓ {name} trained")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Model': name,
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            }
            results.append(metrics)
            
            print(f"\n{name}:")
            for metric, value in metrics.items():
                if metric != 'Model':
                    print(f"  {metric}: {value:.4f}")
        
        results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
        
        return results_df
    
    def fit(self, df):
        """Complete training pipeline"""
        print("="*80)
        print("FRAUD DETECTION TRAINING PIPELINE")
        print("="*80)
        
        # Feature engineering
        df_featured = self.engineer_features(df)
        
        # Preprocessing
        X, y = self.preprocess_data(df_featured)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns
        )
        
        # Apply SMOTE
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"Balanced training samples: {len(X_train_balanced):,}")
        
        # Train models
        self.train_models(X_train_balanced, y_train_balanced)
        
        # Evaluate
        results = self.evaluate_models(X_test_scaled, y_test)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        return results
    
    def predict(self, transaction_data):
        """Predict fraud probability for new transactions"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure correct features
        transaction_df = pd.DataFrame([transaction_data]) if isinstance(transaction_data, dict) else transaction_data
        transaction_df = transaction_df[self.feature_names]
        
        # Scale
        scaled_data = self.scaler.transform(transaction_df)
        
        # Predict
        prediction = self.best_model.predict(scaled_data)[0]
        probability = self.best_model.predict_proba(scaled_data)[0]
        
        return {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability[1]),
            'legitimate_probability': float(probability[0])
        }


def main():
    """Main execution function"""
    print("Loading dataset...")
    # Note: Download dataset from https://www.kaggle.com/datasets/ealaxi/paysim1
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.4f}%")
    
    # Initialize and train pipeline
    pipeline = FraudDetectionPipeline(random_state=RANDOM_STATE)
    results = pipeline.fit(df)
    
    # Save results
    results.to_csv('model_results.csv', index=False)
    print("\n✓ Results saved to model_results.csv")


if __name__ == "__main__":
    main()
  
