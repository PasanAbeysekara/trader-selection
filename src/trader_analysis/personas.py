"""
Personas Module

Trader persona assignment based on trading behavior and performance.
Implements rule-based and statistical persona classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PersonaAssigner:
    """
    Assigns trader personas based on behavioral patterns and performance metrics.
    
    Personas:
    - The Whale: High volume, large position sizes, market moving
    - The Sniper: High win rate, precise entries, low frequency
    - The Scalper: High frequency, small profits, consistent activity
    - The HODLer: Long hold periods, low frequency, patient
    - The Risk Taker: High volatility, large drawdowns, aggressive
    - The Consistent: Steady performance, moderate risk, reliable
    - The Newcomer: Recent activity, limited history
    - The Inactive: No recent activity, dormant
    """
    
    PERSONAS = {
        'The Whale': {
            'description': 'High volume trader with significant market presence',
            'characteristics': 'Large position sizes, substantial capital deployment'
        },
        'The Sniper': {
            'description': 'Precision trader with high win rate and selective entries',
            'characteristics': 'High win rate (>60%), low frequency, excellent timing'
        },
        'The Scalper': {
            'description': 'High-frequency trader capturing small, consistent profits',
            'characteristics': 'Many trades per day, small avg profit, very active'
        },
        'The HODLer': {
            'description': 'Patient long-term holder with infrequent trading',
            'characteristics': 'Low frequency, long holding periods, stable'
        },
        'The Risk Taker': {
            'description': 'Aggressive trader with high risk/reward profile',
            'characteristics': 'High volatility, large drawdowns, boom or bust'
        },
        'The Consistent': {
            'description': 'Reliable trader with steady, predictable performance',
            'characteristics': 'Moderate metrics, consistent activity, low volatility'
        },
        'The Newcomer': {
            'description': 'Recently active trader still establishing track record',
            'characteristics': 'Limited history, recent activity, developing pattern'
        },
        'The Inactive': {
            'description': 'Dormant trader with no recent activity',
            'characteristics': 'No trades in 90+ days, historical data only'
        }
    }
    
    def __init__(self):
        """Initialize PersonaAssigner."""
        self.persona_rules = self._define_persona_rules()
        
    def _define_persona_rules(self) -> Dict:
        """
        Define statistical rules for each persona.
        
        Returns:
        --------
        Dict
            Persona classification rules
        """
        return {
            'The Whale': {
                'volume_percentile': 90,
                'min_total_volume': None,  # Will be calculated from data
                'priority': 1
            },
            'The Sniper': {
                'min_win_rate': 0.60,
                'max_trades_per_day': 2,
                'min_total_trades': 10,
                'min_roi': 0.1,
                'priority': 2
            },
            'The Scalper': {
                'min_trades_per_day': 5,
                'min_active_days': 10,
                'max_avg_win': None,  # Will be calculated
                'priority': 3
            },
            'The HODLer': {
                'max_trades_per_day': 0.5,
                'min_consistency_score': 0.6,
                'min_total_trades': 5,
                'priority': 4
            },
            'The Risk Taker': {
                'volatility_percentile': 80,
                'max_drawdown_percentile': 80,
                'min_total_trades': 10,
                'priority': 5
            },
            'The Consistent': {
                'min_total_trades': 20,
                'min_win_rate': 0.45,
                'max_volatility_percentile': 60,
                'min_consistency_score': 0.5,
                'min_recency_score': 0.3,
                'priority': 6
            },
            'The Newcomer': {
                'max_total_trades': 15,
                'min_recent_trades_30d': 3,
                'max_days_since_last_trade': 30,
                'priority': 7
            },
            'The Inactive': {
                'min_days_since_last_trade': 90,
                'priority': 8
            }
        }
    
    def assign_personas(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign personas to all traders in the dataset.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with all metrics
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with assigned personas and confidence scores
        """
        df = features_df.copy()
        
        # Calculate dynamic thresholds
        volume_threshold = df['total_volume'].quantile(0.90)
        volatility_threshold = df['volatility'].quantile(0.80)
        drawdown_threshold = df['max_drawdown'].quantile(0.20)  # Lower is worse
        avg_win_threshold = df['avg_win'].quantile(0.50)
        
        # Initialize persona columns
        df['persona'] = 'Unassigned'
        df['persona_confidence'] = 0.0
        df['persona_description'] = ''
        
        personas_assigned = []
        
        for idx, row in df.iterrows():
            assigned_persona, confidence = self._classify_trader(
                row, volume_threshold, volatility_threshold, 
                drawdown_threshold, avg_win_threshold
            )
            
            df.at[idx, 'persona'] = assigned_persona
            df.at[idx, 'persona_confidence'] = confidence
            df.at[idx, 'persona_description'] = self.PERSONAS[assigned_persona]['description']
            
            personas_assigned.append(assigned_persona)
        
        # Print summary
        print("\nPersona Assignment Summary:")
        persona_counts = pd.Series(personas_assigned).value_counts()
        for persona, count in persona_counts.items():
            pct = count / len(df) * 100
            print(f"  {persona}: {count} traders ({pct:.1f}%)")
        
        return df
    
    def _classify_trader(self, row: pd.Series, 
                        volume_threshold: float,
                        volatility_threshold: float,
                        drawdown_threshold: float,
                        avg_win_threshold: float) -> Tuple[str, float]:
        """
        Classify a single trader into a persona.
        
        Parameters:
        -----------
        row : pd.Series
            Trader feature row
        volume_threshold : float
            Threshold for whale classification
        volatility_threshold : float
            Threshold for high volatility
        drawdown_threshold : float
            Threshold for large drawdowns
        avg_win_threshold : float
            Threshold for scalper classification
            
        Returns:
        --------
        Tuple[str, float]
            (persona_name, confidence_score)
        """
        # Check personas in priority order
        candidates = []
        
        # The Inactive (highest priority if inactive)
        if row['days_since_last_trade'] >= 90:
            candidates.append(('The Inactive', 0.95))
        
        # The Whale
        if row['total_volume'] >= volume_threshold:
            confidence = min(0.95, row['total_volume'] / volume_threshold * 0.8)
            candidates.append(('The Whale', confidence))
        
        # The Sniper
        if (row['win_rate'] >= 0.60 and 
            row['trades_per_day'] <= 2 and 
            row['total_trades'] >= 10 and
            row['roi'] > 0.1):
            confidence = min(0.95, row['win_rate'] * 1.2)
            candidates.append(('The Sniper', confidence))
        
        # The Scalper
        if (row['trades_per_day'] >= 5 and 
            row['active_days'] >= 10):
            confidence = min(0.90, row['trades_per_day'] / 10 * 0.8)
            candidates.append(('The Scalper', confidence))
        
        # The HODLer
        if (row['trades_per_day'] <= 0.5 and 
            row['total_trades'] >= 5 and
            row.get('consistency_score', 0) >= 0.6):
            confidence = 0.85
            candidates.append(('The HODLer', confidence))
        
        # The Risk Taker
        if (row['volatility'] >= volatility_threshold and 
            row['max_drawdown'] <= drawdown_threshold and
            row['total_trades'] >= 10):
            confidence = 0.85
            candidates.append(('The Risk Taker', confidence))
        
        # The Consistent
        if (row['total_trades'] >= 20 and 
            row['win_rate'] >= 0.45 and
            row['volatility'] < volatility_threshold and
            row.get('consistency_score', 0) >= 0.5 and
            row['recency_score'] >= 0.3):
            confidence = 0.80
            candidates.append(('The Consistent', confidence))
        
        # The Newcomer
        if (row['total_trades'] <= 15 and 
            row['recent_trades_30d'] >= 3 and
            row['days_since_last_trade'] <= 30):
            confidence = 0.75
            candidates.append(('The Newcomer', confidence))
        
        # Select persona with highest confidence
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        
        # Default to Consistent or Newcomer based on activity
        if row['total_trades'] < 20:
            return ('The Newcomer', 0.50)
        else:
            return ('The Consistent', 0.50)
    
    def get_persona_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get aggregated statistics for each persona.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with assigned personas
            
        Returns:
        --------
        pd.DataFrame
            Statistics per persona
        """
        if 'persona' not in features_df.columns:
            raise ValueError("Personas not assigned. Call assign_personas first.")
        
        stats = []
        
        for persona in features_df['persona'].unique():
            persona_data = features_df[features_df['persona'] == persona]
            
            stat = {
                'persona': persona,
                'count': len(persona_data),
                'avg_total_pnl': persona_data['total_pnl'].mean(),
                'avg_roi': persona_data['roi'].mean(),
                'avg_win_rate': persona_data['win_rate'].mean(),
                'avg_sharpe_ratio': persona_data['sharpe_ratio'].mean(),
                'avg_trades': persona_data['total_trades'].mean(),
                'avg_consistency': persona_data.get('consistency_score', pd.Series([0])).mean()
            }
            
            stats.append(stat)
        
        return pd.DataFrame(stats).sort_values('count', ascending=False)
    
    def get_top_traders_by_persona(self, features_df: pd.DataFrame, 
                                   top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Get top N traders for each persona.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with assigned personas
        top_n : int
            Number of top traders to return per persona
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping persona to top traders
        """
        if 'persona' not in features_df.columns:
            raise ValueError("Personas not assigned. Call assign_personas first.")
        
        top_traders = {}
        
        for persona in features_df['persona'].unique():
            persona_data = features_df[features_df['persona'] == persona]
            
            # Sort by weighted PNL and get top N
            top = persona_data.nlargest(top_n, 'weighted_pnl')[
                ['address', 'weighted_pnl', 'roi', 'win_rate', 'total_trades', 'persona_confidence']
            ]
            
            top_traders[persona] = top
        
        return top_traders
    
    def describe_persona(self, persona_name: str) -> Dict:
        """
        Get detailed description of a persona.
        
        Parameters:
        -----------
        persona_name : str
            Name of the persona
            
        Returns:
        --------
        Dict
            Persona details
        """
        if persona_name not in self.PERSONAS:
            raise ValueError(f"Unknown persona: {persona_name}")
        
        return self.PERSONAS[persona_name]
    
    def get_all_personas(self) -> List[str]:
        """Return list of all available personas."""
        return list(self.PERSONAS.keys())
