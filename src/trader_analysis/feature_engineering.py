"""
Feature Engineering Module

Comprehensive feature extraction and engineering for crypto wallet analysis.
Implements recency-weighted metrics and profitability indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Feature engineering for crypto wallet address analysis.
    
    Implements key metrics including:
    - Profitability metrics (ROI, win rate, Sharpe ratio)
    - Trading activity metrics (frequency, volume, consistency)
    - Recency-weighted performance indicators
    - Risk metrics (volatility, drawdown, risk-adjusted returns)
    """
    
    def __init__(self, recency_decay: float = 0.1):
        """
        Initialize FeatureEngineer.
        
        Parameters:
        -----------
        recency_decay : float
            Exponential decay parameter for recency weighting (default: 0.1)
        """
        self.recency_decay = recency_decay
        self.feature_names = []
        
    def calculate_profitability_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate profitability metrics for each wallet address.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: address, timestamp, pnl, entry_price, exit_price
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with profitability metrics per address
        """
        metrics = []
        
        for address, group in df.groupby('address'):
            # Basic profitability metrics
            total_pnl = group['pnl'].sum()
            total_trades = len(group)
            winning_trades = len(group[group['pnl'] > 0])
            losing_trades = len(group[group['pnl'] < 0])
            
            # Win rate and average metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = group[group['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = group[group['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Profit factor
            gross_profit = group[group['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(group[group['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # ROI calculation
            if 'capital_deployed' in group.columns:
                roi = total_pnl / group['capital_deployed'].sum() if group['capital_deployed'].sum() > 0 else 0
            else:
                roi = total_pnl / total_trades if total_trades > 0 else 0
            
            # Sharpe ratio (assuming daily returns)
            returns = group['pnl'].values
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            metrics.append({
                'address': address,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'roi': roi,
                'sharpe_ratio': sharpe_ratio
            })
            
        return pd.DataFrame(metrics)
    
    def calculate_recency_weighted_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recency-weighted performance metrics.
        
        Recent performance is weighted more heavily using exponential decay.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: address, timestamp, pnl
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with recency-weighted metrics per address
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate days since most recent trade
        current_date = df['timestamp'].max()
        df['days_ago'] = (current_date - df['timestamp']).dt.days
        
        # Calculate exponential weights
        df['recency_weight'] = np.exp(-self.recency_decay * df['days_ago'])
        
        metrics = []
        
        for address, group in df.groupby('address'):
            # Weighted PNL
            weighted_pnl = (group['pnl'] * group['recency_weight']).sum()
            
            # Weighted win rate
            weighted_wins = ((group['pnl'] > 0) * group['recency_weight']).sum()
            weighted_total = group['recency_weight'].sum()
            weighted_win_rate = weighted_wins / weighted_total if weighted_total > 0 else 0
            
            # Recent activity score (last 30 days)
            recent_trades = len(group[group['days_ago'] <= 30])
            recent_pnl = group[group['days_ago'] <= 30]['pnl'].sum()
            
            # Days since last trade
            days_since_last_trade = group['days_ago'].min()
            
            metrics.append({
                'address': address,
                'weighted_pnl': weighted_pnl,
                'weighted_win_rate': weighted_win_rate,
                'recent_trades_30d': recent_trades,
                'recent_pnl_30d': recent_pnl,
                'days_since_last_trade': days_since_last_trade,
                'recency_score': 1 / (1 + days_since_last_trade)  # Normalized recency score
            })
            
        return pd.DataFrame(metrics)
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk-adjusted metrics for each wallet.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: address, timestamp, pnl
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with risk metrics per address
        """
        metrics = []
        
        for address, group in df.groupby('address'):
            returns = group['pnl'].values
            
            # Volatility
            volatility = returns.std()
            
            # Maximum drawdown
            cumulative_returns = returns.cumsum()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = drawdown.min()
            
            # Sortino ratio (downside deviation)
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
            
            # Calmar ratio
            calmar_ratio = returns.mean() / abs(max_drawdown) if max_drawdown < 0 else np.inf
            
            # Risk-adjusted return
            risk_adjusted_return = returns.mean() / volatility if volatility > 0 else 0
            
            metrics.append({
                'address': address,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'risk_adjusted_return': risk_adjusted_return
            })
            
        return pd.DataFrame(metrics)
    
    def calculate_trading_activity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading activity and consistency metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with columns: address, timestamp, volume
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with activity metrics per address
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        metrics = []
        
        for address, group in df.groupby('address'):
            # Trading frequency
            total_days = (group['timestamp'].max() - group['timestamp'].min()).days + 1
            trades_per_day = len(group) / total_days if total_days > 0 else 0
            
            # Volume metrics
            if 'volume' in group.columns:
                total_volume = group['volume'].sum()
                avg_volume_per_trade = group['volume'].mean()
            else:
                total_volume = len(group)
                avg_volume_per_trade = 1
            
            # Consistency (coefficient of variation of inter-trade intervals)
            time_diffs = group['timestamp'].diff().dt.total_seconds() / 3600  # hours
            consistency_score = 1 / (1 + time_diffs.std() / time_diffs.mean()) if time_diffs.mean() > 0 else 0
            
            # Active days
            active_days = group['timestamp'].dt.date.nunique()
            
            metrics.append({
                'address': address,
                'trades_per_day': trades_per_day,
                'total_volume': total_volume,
                'avg_volume_per_trade': avg_volume_per_trade,
                'consistency_score': consistency_score,
                'active_days': active_days
            })
            
        return pd.DataFrame(metrics)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to engineer all features for the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw transaction data with columns: address, timestamp, pnl, etc.
            
        Returns:
        --------
        pd.DataFrame
            Comprehensive feature set for each wallet address
        """
        print("Engineering features...")
        
        # Calculate all metric groups
        profitability = self.calculate_profitability_metrics(df)
        recency = self.calculate_recency_weighted_performance(df)
        risk = self.calculate_risk_metrics(df)
        activity = self.calculate_trading_activity_metrics(df)
        
        # Merge all metrics
        features = profitability.merge(recency, on='address', how='outer')
        features = features.merge(risk, on='address', how='outer')
        features = features.merge(activity, on='address', how='outer')
        
        # Fill NaN values
        features = features.fillna(0)
        
        # Replace infinities with large values
        features = features.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Store feature names (excluding address)
        self.feature_names = [col for col in features.columns if col != 'address']
        
        print(f"Engineered {len(self.feature_names)} features for {len(features)} addresses")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        return self.feature_names
