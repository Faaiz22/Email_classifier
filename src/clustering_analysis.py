import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class EmailClustering:
    """
    Clustering analysis for email data to understand patterns in spam/ham emails
    """
    
    def __init__(self):
        self.clusters = {}
        self.dimensionality_reduction = {}
        self.cluster_metrics = {}
        
    def perform_clustering(self, X, method='kmeans', n_clusters=2):
        """
        Perform clustering using specified method
        """
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Fit clustering
        cluster_labels = clusterer.fit_predict(X)
        
        # Calculate silhouette score if more than one cluster
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(X, cluster_labels)
        else:
            silhouette_avg = -1
        
        self.clusters[method] = {
            'clusterer': clusterer,
            'labels': cluster_labels,
            'n_clusters': len(np.unique(cluster_labels)),
            'silhouette_score': silhouette_avg
        }
        
        print(f"{method.upper()} Clustering Results:")
        print(f"Number of clusters: {len(np.unique(cluster_labels))}")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        return cluster_labels, silhouette_avg
    
    def dimensionality_reduction(self, X, method='pca', n_components=2):
        """
        Perform dimensionality reduction for visualization
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Fit and transform
        X_reduced = reducer.fit_transform(X)
        
        self.dimensionality_reduction[method] = {
            'reducer': reducer,
            'transformed_data': X_reduced,
            'explained_variance': getattr(reducer, 'explained_variance_ratio_', None)
        }
        
        if hasattr(reducer, 'explained_variance_ratio_'):
            print(f"{method.upper()} - Explained variance ratio: {reducer.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(reducer.explained_variance_ratio_):.4f}")
        
        return X_reduced
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        """
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow plot
        ax1.plot(cluster_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal Clusters')
        ax1.grid(True)
        
        # Silhouette plot
        ax2.plot(cluster_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal number based on silhouette score
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_clusters}")
        print(f"Best silhouette score: {max(silhouette_scores):.4f}")
        
        return optimal_clusters, inertias, silhouette_scores
    
    def visualize_clusters(self, X, y_true, method='pca', clustering_method='kmeans'):
        """
        Visualize clusters in 2D space
        """
        # Reduce dimensionality
        X_reduced = self.dimensionality_reduction(X, method=method)
        
        # Get cluster labels
        if clustering_method not in self.clusters:
            self.perform_clustering(X, method=clustering_method)
        
        cluster_labels = self.clusters[clustering_method]['labels']
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('True Labels', 'Cluster Labels'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot true labels
        colors_true = ['red' if label == 1 else 'blue' for label in y_true]
        fig.add_trace(
            go.Scatter(
                x=X_reduced[:, 0],
                y=X_reduced[:, 1],
                mode='markers',
                marker=dict(color=colors_true, size=8, opacity=0.7),
                text=[f'True: {"Spam" if label == 1 else "Ham"}' for label in y_true],
                name='True Labels'
            ),
            row=1, col=1
        )
        
        # Plot cluster labels
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set1[:len(unique_clusters)]
        
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            fig.add_trace(
                go.Scatter(
                    x=X_reduced[mask, 0],
                    y=X_reduced[mask, 1],
                    mode='markers',
                    marker=dict(color=colors[i], size=8, opacity=0.7),
                    text=[f'Cluster: {cluster}' for _ in range(sum(mask))],
                    name=f'Cluster {cluster}'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f'Email Clustering Visualization ({method.upper()} + {clustering_method.upper()})',
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text=f'{method.upper()} Component 1')
        fig.update_yaxes(title_text=f'{method.upper()} Component 2')
        
        fig.write_html('clustering_visualization.html')
        fig.show()
        
        return fig
    
    def analyze_cluster_characteristics(self, X, y_true, texts, clustering_method='kmeans'):
        """
        Analyze characteristics of each cluster
        """
        if clustering_method not in self.clusters:
            self.perform_clustering(X, method=clustering_method)
        
        cluster_labels = self.clusters[clustering_method]['labels']
        
        # Create analysis DataFrame
        analysis_df = pd.DataFrame({
            'text': texts,
            'true_label': y_true,
            'cluster': cluster_labels
        })
        
        print("\nCluster Characteristics Analysis")
        print("=" * 50)
        
        for cluster in np.unique(cluster_labels):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster]
            spam_count = sum(cluster_data['true_label'] == 1)
            ham_count = sum(cluster_data['true_label'] == 0)
            total = len(cluster_data)
            
            print(f"\nCluster {cluster} (n={total}):")
            print(f"  Spam emails: {spam_count} ({spam_count/total*100:.1f}%)")
            print(f"  Ham emails: {ham_count} ({ham_count/total*100:.1f}%)")
            
            # Show sample texts
            print("  Sample texts:")
            for i, text in enumerate(cluster_data['text'].head(3)):
                label = "SPAM" if cluster_data.iloc[i]['true_label'] == 1 else "HAM"
                print(f"    [{label}] {text[:80]}...")
        
        # Calculate clustering performance vs true labels
        if len(np.unique(cluster_labels)) == len(np.unique(y_true)):
            ari_score = adjusted_rand_score(y_true, cluster_labels)
            print(f"\nAdjusted Rand Index: {ari_score:.4f}")
            
        return analysis_df
    
    def generate_clustering_report(self, X, y_true, texts):
        """
        Generate comprehensive clustering report
        """
        print("COMPREHENSIVE CLUSTERING ANALYSIS REPORT")
        print("=" * 60)
        
        # Find optimal clusters
        print("\n1. OPTIMAL CLUSTER ANALYSIS")
        print("-" * 30)
        optimal_k, inertias, silhouette_scores = self.find_optimal_clusters(X)
        
        # Perform different clustering methods
        print("\n2. CLUSTERING METHODS COMPARISON")
        print("-" * 40)
        
        methods = ['kmeans', 'dbscan', 'hierarchical']
        for method in methods:
            try:
                self.perform_clustering(X, method=method, n_clusters=optimal_k)
                print()
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # Dimensionality reduction
        print("\n3. DIMENSIONALITY REDUCTION")
        print("-" * 35)
        
        reduction_methods = ['pca', 'tsne', 'svd']
        for method in reduction_methods:
            try:
                print(f"\n{method.upper()}:")
                self.dimensionality_reduction(X, method=method)
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        # Visualization
        print("\n4. VISUALIZATION")
        print("-" * 20)
        try:
            self.visualize_clusters(X, y_true, method='pca', clustering_method='kmeans')
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Cluster characteristics
        print("\n5. CLUSTER CHARACTERISTICS")
        print("-" * 30)
        try:
            self.analyze_cluster_characteristics(X, y_true, texts, clustering_method='kmeans')
        except Exception as e:
            print(f"Analysis error: {e}")
        
        return self.clusters, self.dimensionality_reduction

def perform_email_clustering(data_file='email_dataset.csv'):
    """
    Complete clustering analysis pipeline
    """
    from preprocessor import load_and_preprocess_data
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, processed_df = load_and_preprocess_data(data_file)
    
    # Combine train and test for clustering analysis
    X_combined = np.vstack([X_train.toarray(), X_test.toarray()])
    y_combined = np.hstack([y_train, y_test])
    texts_combined = processed_df['text'].values
    
    # Normalize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Initialize clustering analysis
    clustering = EmailClustering()
    
    # Generate comprehensive report
    clusters, reductions = clustering.generate_clustering_report(X_scaled, y_combined, texts_combined)
    
    return clustering, X_scaled, y_combined, texts_combined

if __name__ == "__main__":
    # Create sample data if needed
    from data_scraper import EmailDataScraper
    
    scraper = EmailDataScraper()
    scraper.scrape_sample_emails()
    df = scraper.create_dataset()
    
    # Perform clustering analysis
    clustering, X_scaled, y_combined, texts = perform_email_clustering()
    
    print("\nClustering analysis complete!")
    print("Check 'clustering_analysis.png' and 'clustering_visualization.html' for visualizations.")