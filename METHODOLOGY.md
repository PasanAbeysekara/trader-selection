# Research Methodology: Crypto Wallet Trader Selection

## Executive Summary

This document outlines the complete research-grade methodology for statistically evaluating and segmenting crypto wallet addresses. The framework combines unsupervised learning (clustering), supervised learning (predictive modeling), and rule-based systems (persona assignment) to identify high-potential traders.

## 1. Problem Statement

**Objective**: Identify and segment crypto wallet addresses that demonstrate consistent profitability and high-potential for future performance.

**Challenges**:
- High dimensionality of trading behavior data
- Class imbalance (few high-performers vs. many average traders)
- Temporal dynamics (recent performance more relevant)
- Noise and volatility in crypto markets
- Need for interpretable results

## 2. Data Requirements

### 2.1 Input Data Schema

Required fields for each transaction:
- `address`: Wallet address (string)
- `timestamp`: Transaction timestamp (datetime)
- `pnl`: Profit/loss for the trade (float)

Optional fields (enhance analysis):
- `volume`: Trade volume/size (float)
- `entry_price`: Entry price (float)
- `exit_price`: Exit price (float)
- `capital_deployed`: Amount invested (float)

### 2.2 Data Quality Requirements

- Minimum 5 trades per wallet address
- At least 30 days of historical data
- Timestamps in chronological order
- No missing values in required fields
- PNL values in consistent currency/units

## 3. Feature Engineering

### 3.1 Profitability Metrics

#### 3.1.1 Total PNL
```
total_pnl = Σ(pnl_i) for all trades i
```

#### 3.1.2 Win Rate
```
win_rate = count(pnl > 0) / total_trades
```

#### 3.1.3 Return on Investment (ROI)
```
ROI = total_pnl / Σ(capital_deployed)
```

#### 3.1.4 Sharpe Ratio (Annualized)
```
Sharpe = (E[returns] / σ[returns]) × √252
```
where E[returns] is mean return and σ[returns] is standard deviation

#### 3.1.5 Profit Factor
```
profit_factor = Σ(pnl | pnl > 0) / |Σ(pnl | pnl < 0)|
```

### 3.2 Recency-Weighted Metrics

#### 3.2.1 Exponential Decay Weighting
```
weight_i = exp(-λ × days_ago_i)
```
where λ = recency_decay parameter (default: 0.1)

#### 3.2.2 Weighted PNL
```
weighted_pnl = Σ(pnl_i × weight_i)
```

#### 3.2.3 Weighted Win Rate
```
weighted_win_rate = Σ((pnl_i > 0) × weight_i) / Σ(weight_i)
```

#### 3.2.4 Recency Score
```
recency_score = 1 / (1 + days_since_last_trade)
```

### 3.3 Risk Metrics

#### 3.3.1 Volatility
```
volatility = σ(returns)
```

#### 3.3.2 Maximum Drawdown
```
max_drawdown = min(cumulative_returns - running_max)
```

#### 3.3.3 Sortino Ratio
```
Sortino = (E[returns] / downside_σ) × √252
```
where downside_σ uses only negative returns

#### 3.3.4 Calmar Ratio
```
Calmar = E[returns] / |max_drawdown|
```

#### 3.3.5 Risk-Adjusted Return
```
risk_adjusted_return = E[returns] / volatility
```

### 3.4 Activity Metrics

#### 3.4.1 Trading Frequency
```
trades_per_day = total_trades / (last_date - first_date).days
```

#### 3.4.2 Consistency Score
```
consistency = 1 / (1 + CV_inter_trade_intervals)
```
where CV is coefficient of variation

#### 3.4.3 Active Days
```
active_days = count(unique(date(timestamp)))
```

## 4. Clustering Analysis

### 4.1 Preprocessing

#### 4.1.1 Feature Scaling
```
X_scaled = (X - μ) / σ
```
Using StandardScaler (zero mean, unit variance)

#### 4.1.2 Handling Outliers
- Replace infinite values with ±10^6
- Winsorize extreme values at 1st and 99th percentiles (optional)

### 4.2 Algorithm Selection

#### 4.2.1 K-Means Clustering
**Advantages**:
- Fast, scalable
- Well-suited for spherical clusters
- Provides clear cluster centers

**Hyperparameters**:
- n_clusters: Optimized using elbow and silhouette methods
- n_init: 10 (multiple initializations)
- max_iter: 300

#### 4.2.2 DBSCAN (Alternative)
**Advantages**:
- Finds arbitrary-shaped clusters
- Identifies noise/outliers
- No need to specify number of clusters

**Hyperparameters**:
- eps: 0.5 (neighborhood radius)
- min_samples: 5 (minimum cluster size)

#### 4.2.3 Hierarchical Clustering (Alternative)
**Advantages**:
- Creates dendrogram for visualization
- No random initialization
- Can capture nested structures

**Hyperparameters**:
- n_clusters: Optimized
- linkage: 'ward' (minimum variance)

### 4.3 Optimal Cluster Selection

#### 4.3.1 Elbow Method
```
elbow_k = argmax(Δ²inertia)
```
Find point of maximum second derivative of inertia curve

#### 4.3.2 Silhouette Analysis
```
silhouette_k = argmax(silhouette_score(X, k))
```
Maximize average silhouette coefficient

#### 4.3.3 Final Selection
```
optimal_k = (elbow_k + silhouette_k) / 2
```

### 4.4 Evaluation Metrics

#### 4.4.1 Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Range: [-1, 1], higher is better

#### 4.4.2 Calinski-Harabasz Index
```
CH = (SSB / (k-1)) / (SSW / (n-k))
```
Higher values indicate better-defined clusters

#### 4.4.3 Davies-Bouldin Index
```
DB = (1/k) × Σ max((σ_i + σ_j) / d(c_i, c_j))
```
Lower values indicate better separation

## 5. Persona Assignment

### 5.1 Persona Definitions

#### 5.1.1 The Whale
**Criteria**:
- total_volume ≥ 90th percentile
- Large capital deployment

**Priority**: 1 (highest)

#### 5.1.2 The Sniper
**Criteria**:
- win_rate ≥ 60%
- trades_per_day ≤ 2
- total_trades ≥ 10
- roi > 10%

**Priority**: 2

#### 5.1.3 The Scalper
**Criteria**:
- trades_per_day ≥ 5
- active_days ≥ 10
- High frequency activity

**Priority**: 3

#### 5.1.4 The HODLer
**Criteria**:
- trades_per_day ≤ 0.5
- consistency_score ≥ 0.6
- total_trades ≥ 5

**Priority**: 4

#### 5.1.5 The Risk Taker
**Criteria**:
- volatility ≥ 80th percentile
- max_drawdown ≥ 80th percentile (magnitude)
- total_trades ≥ 10

**Priority**: 5

#### 5.1.6 The Consistent
**Criteria**:
- total_trades ≥ 20
- win_rate ≥ 45%
- volatility < 60th percentile
- consistency_score ≥ 0.5
- recency_score ≥ 0.3

**Priority**: 6

#### 5.1.7 The Newcomer
**Criteria**:
- total_trades ≤ 15
- recent_trades_30d ≥ 3
- days_since_last_trade ≤ 30

**Priority**: 7

#### 5.1.8 The Inactive
**Criteria**:
- days_since_last_trade ≥ 90

**Priority**: 8 (lowest)

### 5.2 Assignment Algorithm

```python
for each trader:
    candidates = []
    for each persona:
        if trader meets persona criteria:
            confidence = calculate_confidence(trader, persona)
            candidates.append((persona, confidence))
    
    if candidates:
        assigned_persona = max(candidates, key=lambda x: x[1])
    else:
        assigned_persona = default_based_on_activity(trader)
```

## 6. Predictive Modeling

### 6.1 Target Label Creation

#### 6.1.1 High-Potential Criteria
A trader is labeled as high-potential if:
```
(weighted_pnl ≥ top_percentile_threshold) AND
(roi > 0) AND
(total_trades ≥ min_trades) AND
(days_since_last_trade ≤ 60) AND
(win_rate > 0.4)
```

Default: top_percentile = 0.2 (top 20%)

### 6.2 Data Preparation

#### 6.2.1 Train-Test Split
- 80% training, 20% testing
- Stratified sampling to maintain class distribution

#### 6.2.2 SMOTE Oversampling
```
SMOTE(k_neighbors=5, random_state=42)
```
Applied only to training set to balance classes

### 6.3 Ensemble Models

#### 6.3.1 XGBoost
```
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

#### 6.3.2 LightGBM
```
LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

#### 6.3.3 Random Forest
```
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
```

#### 6.3.4 Gradient Boosting
```
GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8
)
```

#### 6.3.5 Logistic Regression (Baseline)
```
LogisticRegression(max_iter=1000)
```

### 6.4 Ensemble Prediction

#### 6.4.1 Soft Voting
```
P(y=1|X) = Σ(w_i × P_i(y=1|X)) / Σ(w_i)
```
where w_i are model weights (default: equal weights)

### 6.5 Evaluation Metrics

#### 6.5.1 Accuracy
```
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### 6.5.2 F1 Score
```
F1 = 2 × (precision × recall) / (precision + recall)
```

#### 6.5.3 ROC AUC
```
AUC = ∫ TPR(FPR) d(FPR)
```

#### 6.5.4 Cross-Validation
5-fold stratified cross-validation on training set

## 7. Statistical Validation

### 7.1 Hypothesis Testing

#### 7.1.1 Mann-Whitney U Test
Compare two independent samples (non-parametric)
```
H0: Distributions are identical
Ha: Distributions differ
α = 0.05
```

#### 7.1.2 Independent t-Test
Compare means of two groups (parametric)
```
t = (μ1 - μ2) / sqrt(s1²/n1 + s2²/n2)
```

#### 7.1.3 Kolmogorov-Smirnov Test
Compare entire distributions
```
D = sup|F1(x) - F2(x)|
```

### 7.2 Effect Size

#### 7.2.1 Cohen's d
```
d = (μ1 - μ2) / pooled_σ
```
Interpretation:
- Small: |d| = 0.2
- Medium: |d| = 0.5
- Large: |d| = 0.8

### 7.3 Confidence Intervals

#### 7.3.1 Parametric (t-distribution)
```
CI = μ ± t_{α/2, df} × (s / √n)
```

#### 7.3.2 Bootstrap (non-parametric)
Resample with replacement B=1000 times

### 7.4 Cluster Stability

#### 7.4.1 Adjusted Rand Index
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```
Range: [-1, 1], where 1 = perfect agreement

Run clustering multiple times and calculate ARI between pairs

## 8. Model Interpretation

### 8.1 Feature Importance

#### 8.1.1 Tree-Based Importance
Average feature importance across tree-based models:
```
importance = mean([
    XGBoost.feature_importances_,
    LightGBM.feature_importances_,
    RandomForest.feature_importances_,
    GradientBoosting.feature_importances_
])
```

### 8.2 Cluster Characterization

For each cluster, compute:
- Mean and median for each feature
- Percentage of total population
- Dominant persona type
- Performance metrics (mean PNL, ROI, win rate)

## 9. Validation Framework

### 9.1 Internal Validation

- **Cross-validation**: 5-fold stratified CV
- **Hold-out test set**: 20% of data
- **Temporal validation**: Train on earlier data, test on recent (if temporal structure available)

### 9.2 External Validation

- **Out-of-sample testing**: Apply to new wallet addresses
- **Robustness checks**: Vary hyperparameters and validate consistency
- **Sensitivity analysis**: Test impact of feature selection

### 9.3 Stability Checks

- **Bootstrap resampling**: Generate confidence intervals
- **Multiple random seeds**: Ensure reproducibility
- **Cluster stability**: ARI across multiple runs

## 10. Recommendations

### 10.1 High-Potential Trader Selection

1. Apply trained ensemble model
2. Rank by predicted probability score
3. Filter by:
   - high_potential_score ≥ 0.7
   - Recent activity (last 30 days)
   - Minimum track record (10+ trades)
4. Select top N traders or top X percentile

### 10.2 Portfolio Construction

Consider diversification across:
- Different personas (e.g., mix Snipers and Consistent traders)
- Different clusters
- Risk levels (combine low and moderate risk)

### 10.3 Monitoring and Updating

- **Retrain frequency**: Every 30 days or after significant market events
- **Performance tracking**: Monitor selected traders' actual performance
- **Model drift detection**: Check if feature distributions change
- **Threshold adjustment**: Recalibrate based on actual outcomes

## 11. Limitations and Considerations

### 11.1 Known Limitations

- **Survivorship bias**: Only includes addresses with transaction history
- **Look-ahead bias**: Ensure no future information leaks into features
- **Market regime changes**: Model trained on specific market conditions
- **Small sample bias**: Some personas may have few representatives

### 11.2 Assumptions

- Past performance indicates (but doesn't guarantee) future results
- Transaction data is complete and accurate
- Wallet addresses represent individual traders (not bots or multiple users)
- PNL calculations are correct and consistent

### 11.3 Best Practices

- Always validate on out-of-sample data
- Monitor model performance in production
- Combine quantitative signals with qualitative assessment
- Update models regularly as new data becomes available
- Document all decisions and parameter choices

## 12. References

### Academic Literature

1. Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python"
2. Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
3. Rousseeuw (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
4. Hubert & Arabie (1985). "Comparing partitions"

### Industry Standards

1. Quantitative Finance: Sharpe ratio, Sortino ratio, Maximum drawdown
2. Machine Learning: ROC AUC, F1 score, cross-validation
3. Statistical Testing: Confidence intervals, hypothesis tests
4. Risk Management: Risk-adjusted returns, diversification

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-14  
**Authors**: MoonCraze Research Team