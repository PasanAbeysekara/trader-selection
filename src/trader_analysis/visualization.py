"""
Visualization Module

Comprehensive visualization tools for trader analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """
    Visualization toolkit for trader analysis.
    
    Provides methods for:
    - Cluster visualization
    - Performance distributions
    - Persona comparisons
    - Feature importance plots
    - Statistical summaries
    """
    
    def __init__(self, output_dir: str = './outputs'):
        """
        Initialize Visualizer.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        self.output_dir = output_dir
        
    def plot_cluster_scatter(self, 
                            features_2d: np.ndarray,
                            labels: np.ndarray,
                            title: str = "Trader Clustering (PCA)",
                            save_path: Optional[str] = None):
        """
        Plot 2D scatter of clusters.
        
        Parameters:
        -----------
        features_2d : np.ndarray
            2D feature array (from PCA)
        labels : np.ndarray
            Cluster labels
        title : str
            Plot title
        save_path : Optional[str]
            Path to save figure
        """
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_persona_distribution(self,
                                  features_df: pd.DataFrame,
                                  save_path: Optional[str] = None):
        """
        Plot distribution of trader personas.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with persona column
        save_path : Optional[str]
            Path to save figure
        """
        if 'persona' not in features_df.columns:
            raise ValueError("Personas not assigned in dataframe")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Count plot
        persona_counts = features_df['persona'].value_counts()
        axes[0].bar(range(len(persona_counts)), persona_counts.values)
        axes[0].set_xticks(range(len(persona_counts)))
        axes[0].set_xticklabels(persona_counts.index, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Traders')
        axes[0].set_title('Trader Persona Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        axes[1].pie(persona_counts.values, labels=persona_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Persona Proportions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_by_persona(self,
                                   features_df: pd.DataFrame,
                                   metric: str = 'total_pnl',
                                   save_path: Optional[str] = None):
        """
        Plot performance metrics by persona.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with persona column
        metric : str
            Metric to plot
        save_path : Optional[str]
            Path to save figure
        """
        if 'persona' not in features_df.columns:
            raise ValueError("Personas not assigned in dataframe")
        
        plt.figure(figsize=(14, 8))
        
        # Box plot
        personas = sorted(features_df['persona'].unique())
        data = [features_df[features_df['persona'] == p][metric].values for p in personas]
        
        bp = plt.boxplot(data, labels=personas, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(personas)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} by Trader Persona')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self,
                               importance_df: pd.DataFrame,
                               top_n: int = 15,
                               save_path: Optional[str] = None):
        """
        Plot feature importance from model.
        
        Parameters:
        -----------
        importance_df : pd.DataFrame
            Dataframe with 'feature' and 'importance' columns
        top_n : int
            Number of top features to show
        save_path : Optional[str]
            Path to save figure
        """
        plt.figure(figsize=(12, 8))
        
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self,
                               correlation_matrix: pd.DataFrame,
                               save_path: Optional[str] = None):
        """
        Plot correlation heatmap.
        
        Parameters:
        -----------
        correlation_matrix : pd.DataFrame
            Correlation matrix
        save_path : Optional[str]
            Path to save figure
        """
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metric_distributions(self,
                                 features_df: pd.DataFrame,
                                 metrics: List[str],
                                 save_path: Optional[str] = None):
        """
        Plot distributions of multiple metrics.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe
        metrics : List[str]
            List of metrics to plot
        save_path : Optional[str]
            Path to save figure
        """
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in features_df.columns:
                axes[idx].hist(features_df[metric].dropna(), bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_xlabel(metric.replace('_', ' ').title())
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'Distribution of {metric.replace("_", " ").title()}')
                axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_top_traders(self,
                        features_df: pd.DataFrame,
                        metric: str = 'weighted_pnl',
                        top_n: int = 20,
                        save_path: Optional[str] = None):
        """
        Plot top N traders by metric.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe
        metric : str
            Metric to rank by
        top_n : int
            Number of top traders
        save_path : Optional[str]
            Path to save figure
        """
        top_traders = features_df.nlargest(top_n, metric)
        
        plt.figure(figsize=(12, 8))
        
        colors = ['green' if x > 0 else 'red' for x in top_traders[metric]]
        
        plt.barh(range(len(top_traders)), top_traders[metric].values, color=colors, alpha=0.7)
        plt.yticks(range(len(top_traders)), 
                  [f"{addr[:6]}...{addr[-4:]}" for addr in top_traders['address'].values])
        plt.xlabel(metric.replace('_', ' ').title())
        plt.title(f'Top {top_n} Traders by {metric.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_summary_dashboard(self,
                                features_df: pd.DataFrame,
                                save_path: Optional[str] = None):
        """
        Create comprehensive summary dashboard.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with all analysis results
        save_path : Optional[str]
            Path to save figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Persona distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'persona' in features_df.columns:
            persona_counts = features_df['persona'].value_counts()
            ax1.bar(range(len(persona_counts)), persona_counts.values)
            ax1.set_xticks(range(len(persona_counts)))
            ax1.set_xticklabels(persona_counts.index, rotation=45, ha='right', fontsize=8)
            ax1.set_title('Persona Distribution')
        
        # 2. PNL distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(features_df['total_pnl'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Total PNL Distribution')
        ax2.set_xlabel('Total PNL')
        
        # 3. Win rate distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(features_df['win_rate'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Win Rate Distribution')
        ax3.set_xlabel('Win Rate')
        
        # 4. ROI distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(features_df['roi'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax4.set_title('ROI Distribution')
        ax4.set_xlabel('ROI')
        
        # 5. Sharpe ratio distribution
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(features_df['sharpe_ratio'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax5.set_title('Sharpe Ratio Distribution')
        ax5.set_xlabel('Sharpe Ratio')
        
        # 6. Trading activity
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(features_df['total_trades'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax6.set_title('Total Trades Distribution')
        ax6.set_xlabel('Total Trades')
        
        # 7. Top traders
        ax7 = fig.add_subplot(gs[2, :])
        top_10 = features_df.nlargest(10, 'weighted_pnl')
        colors = ['green' if x > 0 else 'red' for x in top_10['weighted_pnl']]
        ax7.barh(range(len(top_10)), top_10['weighted_pnl'].values, color=colors, alpha=0.7)
        ax7.set_yticks(range(len(top_10)))
        ax7.set_yticklabels([f"{addr[:8]}..." for addr in top_10['address'].values])
        ax7.set_xlabel('Weighted PNL')
        ax7.set_title('Top 10 Traders by Weighted PNL')
        ax7.invert_yaxis()
        
        plt.suptitle('Trader Analysis Summary Dashboard', fontsize=16, y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
