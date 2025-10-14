"""
Clustering Module

Statistical clustering algorithms for trader segmentation.
Implements multiple clustering approaches with validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


class TraderSegmentation:
    """
    Multi-algorithm clustering system for trader segmentation.
    
    Implements:
    - K-Means clustering
    - DBSCAN (density-based clustering)
    - Hierarchical clustering
    - Automatic optimal cluster selection
    - Statistical validation
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize TraderSegmentation.
        
        Parameters:
        -----------
        n_clusters : int
            Default number of clusters (default: 5)
        random_state : int
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.cluster_centers_ = None
        self.labels_ = None
        self.feature_names = None
        
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Scaled feature matrix
        max_clusters : int
            Maximum number of clusters to test
            
        Returns:
        --------
        int
            Optimal number of clusters
        """
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
        
        # Find elbow point using rate of change
        diffs = np.diff(inertias)
        diff_ratios = diffs[:-1] / diffs[1:]
        elbow_idx = np.argmax(diff_ratios) + 2  # +2 because we start from k=2
        
        # Find best silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores) + 2
        
        # Use average of both methods
        optimal_k = int((elbow_idx + best_silhouette_idx) / 2)
        
        print(f"Elbow method suggests: {elbow_idx} clusters")
        print(f"Silhouette method suggests: {best_silhouette_idx} clusters")
        print(f"Using optimal k: {optimal_k}")
        
        return optimal_k
    
    def fit_kmeans(self, X: np.ndarray, optimize_k: bool = True) -> 'TraderSegmentation':
        """
        Fit K-Means clustering model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        optimize_k : bool
            Whether to automatically find optimal k
            
        Returns:
        --------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_k:
            self.n_clusters = self.find_optimal_clusters(X_scaled)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels_ = self.model.fit_predict(X_scaled)
        self.cluster_centers_ = self.model.cluster_centers_
        
        print(f"K-Means fitted with {self.n_clusters} clusters")
        
        return self
    
    def fit_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> 'TraderSegmentation':
        """
        Fit DBSCAN clustering model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        eps : float
            Maximum distance between samples
        min_samples : int
            Minimum samples in a neighborhood
            
        Returns:
        --------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(X_scaled)
        
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        
        print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return self
    
    def fit_hierarchical(self, X: np.ndarray, optimize_k: bool = True) -> 'TraderSegmentation':
        """
        Fit Hierarchical/Agglomerative clustering model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        optimize_k : bool
            Whether to automatically find optimal k
            
        Returns:
        --------
        self
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize_k:
            self.n_clusters = self.find_optimal_clusters(X_scaled)
        
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'
        )
        
        self.labels_ = self.model.fit_predict(X_scaled)
        
        print(f"Hierarchical clustering fitted with {self.n_clusters} clusters")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_* method first.")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            # For models without predict (like DBSCAN), use nearest cluster center
            return self._predict_nearest_cluster(X_scaled)
    
    def _predict_nearest_cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict cluster by finding nearest existing cluster."""
        # Use the fitted labels to find cluster centers
        cluster_centers = []
        for label in set(self.labels_):
            if label != -1:  # Ignore noise
                cluster_points = X_scaled[self.labels_ == label]
                cluster_centers.append(cluster_points.mean(axis=0))
        
        cluster_centers = np.array(cluster_centers)
        
        # Find nearest cluster for each point
        distances = np.linalg.norm(
            X_scaled[:, np.newaxis] - cluster_centers,
            axis=2
        )
        return np.argmin(distances, axis=1)
    
    def get_cluster_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive statistics for each cluster.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature dataframe with 'address' column
            
        Returns:
        --------
        pd.DataFrame
            Cluster statistics including size, means, and characteristics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit_* method first.")
        
        df = features_df.copy()
        df['cluster'] = self.labels_
        
        # Get feature columns (exclude address and cluster)
        feature_cols = [col for col in df.columns if col not in ['address', 'cluster']]
        
        cluster_stats = []
        
        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = df[df['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Add mean values for each feature
            for col in feature_cols:
                stats[f'mean_{col}'] = cluster_data[col].mean()
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def evaluate_clustering(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit_* method first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Filter out noise points for DBSCAN
        mask = self.labels_ != -1
        X_filtered = X_scaled[mask]
        labels_filtered = self.labels_[mask]
        
        if len(set(labels_filtered)) < 2:
            print("Warning: Less than 2 clusters found, cannot compute some metrics")
            return {}
        
        metrics = {
            'silhouette_score': silhouette_score(X_filtered, labels_filtered),
            'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered),
            'davies_bouldin_score': davies_bouldin_score(X_filtered, labels_filtered),
            'n_clusters': len(set(labels_filtered)),
            'n_noise_points': np.sum(self.labels_ == -1)
        }
        
        print("\nClustering Evaluation Metrics:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better)")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
        print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        print(f"  Number of Clusters: {metrics['n_clusters']}")
        print(f"  Noise Points: {metrics['n_noise_points']}")
        
        return metrics
    
    def reduce_dimensions_for_visualization(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions using PCA for visualization.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        n_components : int
            Number of components for PCA
            
        Returns:
        --------
        np.ndarray
            Reduced dimension features
        """
        X_scaled = self.scaler.transform(X)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pca.fit_transform(X_scaled)
