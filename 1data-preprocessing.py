"""
Data Preprocessing and Machine Learning Implementation
For IBKR Automated Trading System Research
Author: YUQIONG TONG
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ====================
# DATA PREPROCESSING
# ====================

class IBKRDataProcessor:
    """Process Interactive Brokers activity statements and trading data"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.target = None
        
    def load_data(self):
        """Load trading data from Excel file"""
        print("Loading IBKR trading data...")
        self.raw_data = pd.read_excel(self.file_path, sheet_name='Table 1')
        
        # Clean column names
        self.raw_data.columns = [col.strip() for col in self.raw_data.columns]
        self.raw_data = self.raw_data.rename(columns={
            'Date/Time': 'datetime',
            'quantatative': 'quantity',
            'Transaction Price': 'transaction_price',
            'Closing Price': 'closing_price',
            'Profit': 'profit',
            'Commission/Tax': 'commission',
            'Cost Basis': 'cost_basis',
            'Realized Profit and Loss': 'realized_pnl',
            'Mark-to-Market  Profit and Loss': 'mtm_pnl'
        })
        
        print(f"Loaded {len(self.raw_data)} trading records")
        return self.raw_data
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        df = self.raw_data.copy()
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'].str.replace('\n', ' '))
        
        # Extract time features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_us_market_hours'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        
        # Calculate additional features
        df['position_value'] = abs(df['quantity'] * df['transaction_price'])
        df['price_change'] = df['closing_price'] - df['transaction_price']
        df['price_change_pct'] = (df['price_change'] / df['transaction_price']) * 100
        df['commission_rate'] = abs(df['commission'] / df['position_value'])
        
        # Volatility proxy (rolling standard deviation of price changes)
        df = df.sort_values('datetime')
        df['price_volatility'] = df.groupby('stock')['price_change_pct'].transform(
            lambda x: x.rolling(window=20, min_periods=1).std()
        )
        
        # Risk score based on volatility and position size
        df['risk_score'] = df['price_volatility'] * np.log1p(df['position_value'])
        
        # Profit/Loss indicators
        df['is_profitable'] = (df['mtm_pnl'] > 0).astype(int)
        
        self.processed_data = df
        print(f"Preprocessing complete. Shape: {df.shape}")
        return df
    
    def engineer_features(self):
        """Create advanced features for ML models"""
        df = self.processed_data.copy()
        
        # Technical indicators
        for stock in df['stock'].unique():
            stock_data = df[df['stock'] == stock].sort_values('datetime')
            
            # Moving averages
            stock_data['ma_5'] = stock_data['transaction_price'].rolling(5).mean()
            stock_data['ma_20'] = stock_data['transaction_price'].rolling(20).mean()
            
            # Price relative to MA
            stock_data['price_to_ma5'] = stock_data['transaction_price'] / stock_data['ma_5']
            stock_data['price_to_ma20'] = stock_data['transaction_price'] / stock_data['ma_20']
            
            # Volume patterns (using quantity as proxy)
            stock_data['volume_ma'] = stock_data['quantity'].abs().rolling(10).mean()
            stock_data['volume_ratio'] = stock_data['quantity'].abs() / stock_data['volume_ma']
            
            df.loc[df['stock'] == stock, stock_data.columns] = stock_data
        
        # Lag features
        df['prev_pnl'] = df.groupby('stock')['mtm_pnl'].shift(1)
        df['prev_volatility'] = df.groupby('stock')['price_volatility'].shift(1)
        
        # Interaction features
        df['volatility_position_interaction'] = df['price_volatility'] * df['position_value']
        df['risk_adjusted_position'] = df['position_value'] / (1 + df['price_volatility'])
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        self.processed_data = df
        
        # Select features for modeling
        self.features = df[[
            'quantity', 'transaction_price', 'position_value', 'price_change_pct',
            'commission_rate', 'price_volatility', 'risk_score', 'hour', 'day_of_week',
            'is_us_market_hours', 'price_to_ma5', 'price_to_ma20', 'volume_ratio',
            'prev_pnl', 'prev_volatility', 'volatility_position_interaction'
        ]]
        
        self.target = df['is_profitable']
        
        print(f"Feature engineering complete. {len(self.features.columns)} features created")
        return self.features, self.target

# ====================
# MACHINE LEARNING MODELS
# ====================

class TradingMLModels:
    """Machine Learning models for trading pattern identification"""
    
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, test_size=0.2):
        """Prepare data for training"""
        # Scale features
        X_scaled = self.scaler.fit_transform(self.features)
        
        # Time series split for financial data
        split_idx = int(len(X_scaled) * (1 - test_size))
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = self.target.iloc[:split_idx]
        y_test = self.target.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {'accuracy': accuracy}
        
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {'accuracy': accuracy}
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        return model
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        """Train Decision Tree for interpretability"""
        print("Training Decision Tree model...")
        
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['decision_tree'] = model
        self.results['decision_tree'] = {'accuracy': accuracy}
        
        print(f"Decision Tree Accuracy: {accuracy:.4f}")
        return model
    
    def perform_clustering(self, n_clusters=4):
        """Perform clustering analysis to identify trading patterns"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        X_scaled = self.scaler.fit_transform(self.features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_df = pd.DataFrame(self.features)
        cluster_df['cluster'] = clusters
        cluster_df['profitable'] = self.target
        
        cluster_summary = cluster_df.groupby('cluster').agg({
            'position_value': 'mean',
            'price_volatility': 'mean',
            'risk_score': 'mean',
            'profitable': 'mean'
        }).round(3)
        
        cluster_summary['count'] = cluster_df.groupby('cluster').size()
        cluster_summary['percentage'] = (cluster_summary['count'] / len(cluster_df) * 100).round(1)
        
        self.models['kmeans'] = kmeans
        
        print("\nCluster Analysis:")
        print(cluster_summary)
        
        # Assign cluster names based on characteristics
        cluster_names = {
            0: "Conservative",
            1: "High-Risk",
            2: "Large Cap",
            3: "Speculative"
        }
        
        return clusters, cluster_summary, cluster_names

# ====================
# ANALYSIS & VALIDATION
# ====================

class TradingAnalysis:
    """Statistical analysis and validation of trading patterns"""
    
    def __init__(self, processed_data):
        self.data = processed_data
        
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\nPerforming correlation analysis...")
        
        correlation_vars = [
            'price_change_pct', 'mtm_pnl', 'position_value', 
            'price_volatility', 'risk_score'
        ]
        
        corr_matrix = self.data[correlation_vars].corr()
        
        # Key correlations
        print("\nKey Correlations with P&L:")
        pnl_corr = corr_matrix['mtm_pnl'].sort_values(ascending=False)
        for var, corr in pnl_corr.items():
            if var != 'mtm_pnl':
                print(f"  {var}: {corr:.3f}")
        
        return corr_matrix
    
    def volatility_position_analysis(self):
        """Analyze volatility-position size interaction"""
        print("\nAnalyzing volatility-position size interaction...")
        
        # Create bins for analysis
        self.data['volatility_bin'] = pd.qcut(self.data['price_volatility'], q=3, 
                                               labels=['Low', 'Medium', 'High'])
        self.data['position_bin'] = pd.qcut(self.data['position_value'], q=3,
                                            labels=['Small', 'Medium', 'Large'])
        
        # Cross-tabulation
        profitability_matrix = pd.crosstab(
            self.data['volatility_bin'],
            self.data['position_bin'],
            self.data['is_profitable'],
            aggfunc='mean'
        ).round(3)
        
        print("\nProfitability by Volatility and Position Size:")
        print(profitability_matrix)
        
        return profitability_matrix
    
    def execution_quality_analysis(self):
        """Analyze execution quality factors"""
        print("\nAnalyzing execution quality...")
        
        # Calculate slippage proxy
        self.data['slippage'] = abs(self.data['transaction_price'] - self.data['closing_price'])
        self.data['slippage_pct'] = (self.data['slippage'] / self.data['transaction_price']) * 100
        
        # Execution quality metrics
        metrics = {
            'avg_slippage_pct': self.data['slippage_pct'].mean(),
            'avg_commission_rate': self.data['commission_rate'].mean(),
            'total_commission': self.data['commission'].sum(),
            'execution_cost_impact': (self.data['commission'].sum() / self.data['profit'].sum()) * 100
        }
        
        print("\nExecution Quality Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def strategy_performance_analysis(self):
        """Analyze performance by different strategies"""
        print("\nAnalyzing strategy performance...")
        
        # Define strategy conditions
        strategies = {
            'MSTY_Volatility': (self.data['price_volatility'] > self.data['price_volatility'].median()) & 
                               (self.data['volume_ratio'] > 1),
            'Low_Vol_Large': (self.data['price_volatility'] < self.data['price_volatility'].quantile(0.33)) &
                            (self.data['position_value'] > self.data['position_value'].quantile(0.67)),
            'Tech_Pattern': (self.data['stock'].isin(['AAPL', 'GOOGL', 'MSFT', 'NVDA'])),
            'Forex_Hedge': (self.data['hour'] >= 22) | (self.data['hour'] <= 6)
        }
        
        strategy_results = {}
        for name, condition in strategies.items():
            strategy_data = self.data[condition]
            if len(strategy_data) > 0:
                strategy_results[name] = {
                    'count': len(strategy_data),
                    'win_rate': strategy_data['is_profitable'].mean(),
                    'avg_return': strategy_data['mtm_pnl'].mean(),
                    'risk': strategy_data['risk_score'].mean()
                }
        
        results_df = pd.DataFrame(strategy_results).T
        print("\nStrategy Performance:")
        print(results_df)
        
        return results_df

# ====================
# MAIN EXECUTION
# ====================

def main():
    """Main execution function"""
    print("="*60)
    print("IBKR Automated Trading System - ML Implementation")
    print("="*60)
    
    # 1. Data Processing
    processor = IBKRDataProcessor('Trading Activity datasets.xlsx')
    raw_data = processor.load_data()
    processed_data = processor.preprocess_data()
    features, target = processor.engineer_features()
    
    # 2. Machine Learning Models
    ml_models = TradingMLModels(features, target)
    X_train, X_test, y_train, y_test = ml_models.prepare_data()
    
    # Train models
    xgb_model, feature_importance = ml_models.train_xgboost(X_train, y_train, X_test, y_test)
    rf_model = ml_models.train_random_forest(X_train, y_train, X_test, y_test)
    dt_model = ml_models.train_decision_tree(X_train, y_train, X_test, y_test)
    
    # Clustering
    clusters, cluster_summary, cluster_names = ml_models.perform_clustering()
    
    # 3. Analysis
    analysis = TradingAnalysis(processed_data)
    corr_matrix = analysis.correlation_analysis()
    prof_matrix = analysis.volatility_position_analysis()
    exec_metrics = analysis.execution_quality_analysis()
    strategy_perf = analysis.strategy_performance_analysis()
    
    # 4. Feature Importance
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # 5. Model Comparison
    print("\nModel Performance Summary:")
    for model_name, results in ml_models.results.items():
        print(f"  {model_name}: Accuracy = {results['accuracy']:.4f}")
    
    # Save models and results
    import joblib
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(processor.scaler, 'feature_scaler.pkl')
    processed_data.to_csv('processed_trading_data.csv', index=False)
    
    print("\n" + "="*60)
    print("Analysis Complete! Models and results saved.")
    print("="*60)
    
    return processor, ml_models, analysis

if __name__ == "__main__":
    processor, ml_models, analysis = main()
