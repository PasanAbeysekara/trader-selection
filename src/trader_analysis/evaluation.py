"""
Evaluation Module

Statistical validation and evaluation techniques for trader analysis models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import silhouette_samples
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation and statistical validation framework.
    
    Implements:
    - Statistical significance testing
    - Model stability analysis
    - Backtesting framework
    - Performance benchmarking
    - Cross-validation statistics
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize ModelEvaluator.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level for statistical tests (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def evaluate_cluster_stability(self, 
                                   labels_list: List[np.ndarray],
                                   features: np.ndarray) -> Dict:
        """
        Evaluate stability of clustering across multiple runs.
        
        Parameters:
        -----------
        labels_list : List[np.ndarray]
            List of cluster labels from multiple runs
        features : np.ndarray
            Feature matrix
            
        Returns:
        --------
        Dict
            Stability metrics
        """
        if len(labels_list) < 2:
            raise ValueError("Need at least 2 clustering runs for stability analysis")
        
        # Calculate adjusted rand index between pairs
        from sklearn.metrics import adjusted_rand_score
        
        ari_scores = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                ari_scores.append(ari)
        
        stability = {
            'mean_ari': np.mean(ari_scores),
            'std_ari': np.std(ari_scores),
            'min_ari': np.min(ari_scores),
            'max_ari': np.max(ari_scores)
        }
        
        print("\nCluster Stability Analysis:")
        print(f"  Mean Adjusted Rand Index: {stability['mean_ari']:.4f}")
        print(f"  Std Adjusted Rand Index: {stability['std_ari']:.4f}")
        print(f"  Interpretation: {'Highly stable' if stability['mean_ari'] > 0.8 else 'Moderately stable' if stability['mean_ari'] > 0.6 else 'Low stability'}")
        
        return stability
    
    def statistical_comparison(self, 
                              group1: np.ndarray, 
                              group2: np.ndarray,
                              test: str = 'ttest') -> Dict:
        """
        Perform statistical test to compare two groups.
        
        Parameters:
        -----------
        group1 : np.ndarray
            First group values
        group2 : np.ndarray
            Second group values
        test : str
            Statistical test to use ('ttest', 'mannwhitney', 'ks')
            
        Returns:
        --------
        Dict
            Test results including statistic and p-value
        """
        if test == 'ttest':
            statistic, pvalue = stats.ttest_ind(group1, group2)
            test_name = "Independent T-Test"
        elif test == 'mannwhitney':
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U Test"
        elif test == 'ks':
            statistic, pvalue = stats.ks_2samp(group1, group2)
            test_name = "Kolmogorov-Smirnov Test"
        else:
            raise ValueError(f"Unknown test: {test}")
        
        significant = pvalue < self.alpha
        
        results = {
            'test_name': test_name,
            'statistic': statistic,
            'pvalue': pvalue,
            'significant': significant,
            'alpha': self.alpha,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'effect_size': (np.mean(group1) - np.mean(group2)) / np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
        }
        
        return results
    
    def compare_personas(self, features_df: pd.DataFrame, 
                        metric: str = 'total_pnl') -> pd.DataFrame:
        """
        Statistically compare performance across personas.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with assigned personas
        metric : str
            Metric to compare (default: 'total_pnl')
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        if 'persona' not in features_df.columns:
            raise ValueError("Personas not assigned in dataframe")
        
        personas = features_df['persona'].unique()
        results = []
        
        for i, persona1 in enumerate(personas):
            for persona2 in personas[i+1:]:
                group1 = features_df[features_df['persona'] == persona1][metric].values
                group2 = features_df[features_df['persona'] == persona2][metric].values
                
                if len(group1) > 1 and len(group2) > 1:
                    comparison = self.statistical_comparison(group1, group2, test='mannwhitney')
                    
                    results.append({
                        'persona1': persona1,
                        'persona2': persona2,
                        'metric': metric,
                        'persona1_mean': comparison['group1_mean'],
                        'persona2_mean': comparison['group2_mean'],
                        'pvalue': comparison['pvalue'],
                        'significant': comparison['significant'],
                        'effect_size': comparison['effect_size']
                    })
        
        results_df = pd.DataFrame(results)
        
        print(f"\nStatistical Comparison of Personas ({metric}):")
        print(f"Significant differences (p < {self.alpha}):")
        sig_results = results_df[results_df['significant']]
        for _, row in sig_results.iterrows():
            print(f"  {row['persona1']} vs {row['persona2']}: p={row['pvalue']:.4f}, effect_size={row['effect_size']:.4f}")
        
        return results_df
    
    def calculate_confidence_intervals(self, 
                                      features_df: pd.DataFrame,
                                      metric: str = 'total_pnl') -> pd.DataFrame:
        """
        Calculate confidence intervals for each persona.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with assigned personas
        metric : str
            Metric to analyze
            
        Returns:
        --------
        pd.DataFrame
            Confidence intervals per persona
        """
        if 'persona' not in features_df.columns:
            raise ValueError("Personas not assigned in dataframe")
        
        results = []
        
        for persona in features_df['persona'].unique():
            data = features_df[features_df['persona'] == persona][metric].values
            
            if len(data) > 1:
                mean = np.mean(data)
                sem = stats.sem(data)
                ci = stats.t.interval(self.confidence_level, len(data)-1, loc=mean, scale=sem)
                
                results.append({
                    'persona': persona,
                    'metric': metric,
                    'mean': mean,
                    'std': np.std(data),
                    'sem': sem,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'n': len(data)
                })
        
        return pd.DataFrame(results)
    
    def backtest_prediction_model(self,
                                 features_df: pd.DataFrame,
                                 predictor,
                                 n_periods: int = 5) -> Dict:
        """
        Perform time-series backtesting for prediction model.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with timestamp information
        predictor : HighPotentialPredictor
            Trained predictor model
        n_periods : int
            Number of time periods for backtesting
            
        Returns:
        --------
        Dict
            Backtesting results
        """
        # This would require temporal data structure
        # Placeholder for demonstration
        print("\nBacktesting framework:")
        print("  Requires temporal dataset with multiple time periods")
        print("  Would implement walk-forward validation")
        print("  Would track prediction accuracy over time")
        
        return {
            'message': 'Backtest requires temporal data structure',
            'periods': n_periods
        }
    
    def calculate_sharpe_ratio_significance(self,
                                           returns: np.ndarray,
                                           benchmark_sharpe: float = 0) -> Dict:
        """
        Test statistical significance of Sharpe ratio.
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        benchmark_sharpe : float
            Benchmark Sharpe ratio for comparison
            
        Returns:
        --------
        Dict
            Statistical test results
        """
        n = len(returns)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Standard error of Sharpe ratio
        sharpe_se = np.sqrt((1 + 0.5 * sharpe**2) / n)
        
        # Z-score
        z_score = (sharpe - benchmark_sharpe) / sharpe_se
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        significant = pvalue < self.alpha
        
        results = {
            'sharpe_ratio': sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'standard_error': sharpe_se,
            'z_score': z_score,
            'pvalue': pvalue,
            'significant': significant,
            'interpretation': 'Significantly better than benchmark' if significant and sharpe > benchmark_sharpe else 'Not significantly different'
        }
        
        return results
    
    def perform_feature_correlation_analysis(self, 
                                            features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze feature correlations and identify multicollinearity.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        # Select numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'address']
        
        correlation_matrix = features_df[numeric_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print("\nHighly Correlated Features (|r| > 0.8):")
            for pair in high_corr_pairs:
                print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("\nNo highly correlated feature pairs found (|r| > 0.8)")
        
        return correlation_matrix
    
    def calculate_portfolio_metrics(self, 
                                    selected_traders_df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level metrics for selected traders.
        
        Parameters:
        -----------
        selected_traders_df : pd.DataFrame
            Dataframe of selected high-potential traders
            
        Returns:
        --------
        Dict
            Portfolio metrics
        """
        metrics = {
            'total_traders': len(selected_traders_df),
            'total_pnl': selected_traders_df['total_pnl'].sum(),
            'avg_roi': selected_traders_df['roi'].mean(),
            'avg_sharpe': selected_traders_df['sharpe_ratio'].mean(),
            'avg_win_rate': selected_traders_df['win_rate'].mean(),
            'total_trades': selected_traders_df['total_trades'].sum(),
            'persona_distribution': selected_traders_df['persona'].value_counts().to_dict() if 'persona' in selected_traders_df.columns else {}
        }
        
        print("\nPortfolio Metrics:")
        print(f"  Selected Traders: {metrics['total_traders']}")
        print(f"  Total PNL: {metrics['total_pnl']:.2f}")
        print(f"  Average ROI: {metrics['avg_roi']:.4f}")
        print(f"  Average Sharpe: {metrics['avg_sharpe']:.4f}")
        print(f"  Average Win Rate: {metrics['avg_win_rate']:.4f}")
        
        return metrics
    
    def generate_evaluation_report(self,
                                  features_df: pd.DataFrame,
                                  clustering_metrics: Dict,
                                  prediction_metrics: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with all analysis results
        clustering_metrics : Dict
            Clustering evaluation metrics
        prediction_metrics : Optional[Dict]
            Prediction model metrics
            
        Returns:
        --------
        Dict
            Comprehensive evaluation report
        """
        report = {
            'dataset_summary': {
                'total_addresses': len(features_df),
                'total_trades': features_df['total_trades'].sum(),
                'total_pnl': features_df['total_pnl'].sum(),
                'date_range': 'Not available in features'
            },
            'clustering_evaluation': clustering_metrics,
            'persona_distribution': features_df['persona'].value_counts().to_dict() if 'persona' in features_df.columns else {},
            'prediction_evaluation': prediction_metrics if prediction_metrics else {},
            'top_performers': {
                'by_pnl': features_df.nlargest(10, 'weighted_pnl')[['address', 'weighted_pnl', 'persona']].to_dict('records') if 'persona' in features_df.columns else [],
                'by_roi': features_df.nlargest(10, 'roi')[['address', 'roi', 'persona']].to_dict('records') if 'persona' in features_df.columns else []
            }
        }
        
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        print(f"\nDataset Summary:")
        print(f"  Total Addresses Analyzed: {report['dataset_summary']['total_addresses']}")
        print(f"  Total Trades: {report['dataset_summary']['total_trades']}")
        print(f"  Total PNL: {report['dataset_summary']['total_pnl']:.2f}")
        
        return report
