"""
Sample Data Generator

Generates synthetic crypto wallet transaction data for demonstration purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_sample_wallet_data(
    n_wallets: int = 100,
    n_trades_range: tuple = (5, 200),
    date_range_days: int = 365,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic wallet trading data.
    
    Parameters:
    -----------
    n_wallets : int
        Number of wallet addresses to generate
    n_trades_range : tuple
        Min and max number of trades per wallet
    date_range_days : int
        Number of days of historical data
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Synthetic transaction data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=date_range_days)
    
    # Generate wallet addresses
    wallets = [f"0x{random.randbytes(20).hex()}" for _ in range(n_wallets)]
    
    # Define trader archetypes with different behaviors
    archetypes = {
        'whale': {'prob': 0.05, 'volume_mult': 10, 'win_rate': 0.55, 'frequency': 0.3},
        'sniper': {'prob': 0.10, 'volume_mult': 1, 'win_rate': 0.70, 'frequency': 0.1},
        'scalper': {'prob': 0.15, 'volume_mult': 0.5, 'win_rate': 0.52, 'frequency': 3.0},
        'hodler': {'prob': 0.10, 'volume_mult': 1.5, 'win_rate': 0.50, 'frequency': 0.05},
        'risk_taker': {'prob': 0.15, 'volume_mult': 2, 'win_rate': 0.45, 'frequency': 0.8},
        'consistent': {'prob': 0.25, 'volume_mult': 1, 'win_rate': 0.53, 'frequency': 0.5},
        'newcomer': {'prob': 0.10, 'volume_mult': 0.8, 'win_rate': 0.48, 'frequency': 0.2},
        'inactive': {'prob': 0.10, 'volume_mult': 1, 'win_rate': 0.50, 'frequency': 0.01}
    }
    
    for wallet in wallets:
        # Assign archetype
        archetype_name = random.choices(
            list(archetypes.keys()),
            weights=[a['prob'] for a in archetypes.values()]
        )[0]
        archetype = archetypes[archetype_name]
        
        # Determine number of trades
        n_trades = random.randint(*n_trades_range)
        
        # Adjust based on archetype frequency
        n_trades = int(n_trades * archetype['frequency'])
        n_trades = max(1, n_trades)  # At least 1 trade
        
        # Generate trades for this wallet
        wallet_capital = np.random.uniform(1000, 100000)
        
        for _ in range(n_trades):
            # Generate timestamp
            # More recent for active traders, older for inactive
            if archetype_name == 'inactive':
                days_ago = random.randint(90, date_range_days)
            elif archetype_name == 'newcomer':
                days_ago = random.randint(0, 60)
            else:
                days_ago = random.randint(0, date_range_days)
            
            timestamp = end_date - timedelta(days=days_ago)
            
            # Generate trade details
            position_size = wallet_capital * np.random.uniform(0.05, 0.3) * archetype['volume_mult']
            
            # Determine win/loss based on archetype win rate
            is_win = random.random() < archetype['win_rate']
            
            if is_win:
                # Winning trade
                profit_pct = np.random.uniform(0.02, 0.15)
                pnl = position_size * profit_pct
            else:
                # Losing trade
                loss_pct = np.random.uniform(-0.10, -0.01)
                pnl = position_size * loss_pct
            
            # Add some noise to make it more realistic
            pnl *= np.random.uniform(0.8, 1.2)
            
            # Entry and exit prices (simplified)
            entry_price = np.random.uniform(100, 50000)
            exit_price = entry_price * (1 + pnl / position_size)
            
            data.append({
                'address': wallet,
                'timestamp': timestamp,
                'pnl': pnl,
                'volume': position_size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'capital_deployed': position_size,
                'true_archetype': archetype_name  # For validation
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(['address', 'timestamp']).reset_index(drop=True)
    
    print(f"Generated {len(df)} trades for {n_wallets} wallets")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nArchetype distribution:")
    archetype_dist = df.groupby('address')['true_archetype'].first().value_counts()
    for arch, count in archetype_dist.items():
        print(f"  {arch}: {count} wallets ({count/n_wallets*100:.1f}%)")
    
    return df


if __name__ == '__main__':
    # Generate sample data
    df = generate_sample_wallet_data(
        n_wallets=200,
        n_trades_range=(5, 300),
        date_range_days=365,
        seed=42
    )
    
    # Save to CSV
    output_path = '../data/raw/sample_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSample data saved to: {output_path}")
