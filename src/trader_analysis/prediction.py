"""
Prediction Module

Predictive modeling for identifying high-potential traders.
Implements ensemble methods and probability calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class HighPotentialPredictor:
    """
    Ensemble predictor for identifying high-potential traders.
    
    Implements:
    - Multiple ML algorithms (XGBoost, LightGBM, Random Forest)
    - Ensemble voting system
    - SMOTE for class imbalance
    - Cross-validation
    - Feature importance analysis
    """
    
    def __init__(self, random_state: int = 42, use_smote: bool = True):
        """
        Initialize HighPotentialPredictor.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        use_smote : bool
            Whether to use SMOTE for handling class imbalance
        """
        self.random_state = random_state
        self.use_smote = use_smote
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importances_ = None
        self.feature_names = None
        
    def create_target_labels(self, features_df: pd.DataFrame, 
                            top_percentile: float = 0.2,
                            min_trades: int = 10) -> pd.Series:
        """
        Create binary target labels for high-potential traders.
        
        High-potential defined as:
        - Top percentile in weighted PNL
        - Positive ROI
        - Minimum number of trades
        - Recent activity (traded in last 60 days)
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe
        top_percentile : float
            Top percentile threshold (default: 0.2 for top 20%)
        min_trades : int
            Minimum number of trades required
            
        Returns:
        --------
        pd.Series
            Binary labels (1 = high potential, 0 = not high potential)
        """
        df = features_df.copy()
        
        # Calculate threshold for top percentile
        pnl_threshold = df['weighted_pnl'].quantile(1 - top_percentile)
        
        # Define high-potential criteria
        high_potential = (
            (df['weighted_pnl'] >= pnl_threshold) &
            (df['roi'] > 0) &
            (df['total_trades'] >= min_trades) &
            (df['days_since_last_trade'] <= 60) &
            (df['win_rate'] > 0.4)
        )
        
        labels = high_potential.astype(int)
        
        print(f"Created labels: {labels.sum()} high-potential traders ({labels.sum()/len(labels)*100:.1f}%)")
        
        return labels
    
    def prepare_data(self, features_df: pd.DataFrame, 
                    target: pd.Series,
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training including scaling and SMOTE.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe
        target : pd.Series
            Target labels
        test_size : float
            Proportion of test set
            
        Returns:
        --------
        Tuple
            (X_train, X_test, y_train, y_test)
        """
        # Get feature columns (exclude address)
        feature_cols = [col for col in features_df.columns if col != 'address']
        X = features_df[feature_cols].values
        y = target.values
        
        self.feature_names = feature_cols
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE to training data
        if self.use_smote and len(np.unique(y_train)) > 1:
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"SMOTE applied: Training set size increased to {len(y_train)} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train ensemble of models.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
            
        Returns:
        --------
        Dict
            Dictionary of trained models
        """
        print("\nTraining ensemble models...")
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train, y_train)
        print("  ✓ XGBoost trained")
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train, y_train)
        print("  ✓ LightGBM trained")
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )
        self.models['random_forest'].fit(X_train, y_train)
        print("  ✓ Random Forest trained")
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state
        )
        self.models['gradient_boosting'].fit(X_train, y_train)
        print("  ✓ Gradient Boosting trained")
        
        # Logistic Regression (as baseline)
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.models['logistic_regression'].fit(X_train, y_train)
        print("  ✓ Logistic Regression trained")
        
        return self.models
    
    def predict_proba_ensemble(self, X: np.ndarray, weights: Optional[Dict] = None) -> np.ndarray:
        """
        Predict probabilities using ensemble voting.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        weights : Optional[Dict]
            Weights for each model (if None, equal weights)
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        if not self.models:
            raise ValueError("Models not trained yet. Call train_ensemble first.")
        
        if weights is None:
            weights = {name: 1.0 for name in self.models.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Get predictions from each model
        predictions = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)
            predictions += pred_proba * weights.get(name, 0)
        
        return predictions
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using ensemble.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        threshold : float
            Probability threshold for positive class
            
        Returns:
        --------
        np.ndarray
            Predicted class labels
        """
        proba = self.predict_proba_ensemble(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        Dict
            Evaluation metrics
        """
        print("\nEvaluating ensemble performance...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba_ensemble(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\nEnsemble Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Not High-Potential', 'High-Potential']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance from tree-based models.
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.models or self.feature_names is None:
            raise ValueError("Models not trained yet.")
        
        importances = {}
        
        # Get importances from tree-based models
        for name in ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']:
            if name in self.models:
                importances[name] = self.models[name].feature_importances_
        
        # Calculate average importance
        avg_importance = np.mean(list(importances.values()), axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation on the ensemble.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict
            Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_scores = {}
        
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
            cv_scores[name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
            print(f"  {name}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_scores
