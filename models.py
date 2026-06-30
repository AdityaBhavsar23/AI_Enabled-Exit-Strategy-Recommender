"""
AI/ML Models for Exit Strategy Prediction
Contains Random Forest, Gradient Boosting, and K-Means Clustering
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ExitStrategyAI:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.strategies = ["IPO", "Strategic Sale", "Merger", "Buyback", "Private Equity Exit", "Secondary Sale"]
        
        # Initialize or load models
        self.init_models()
    
    def init_models(self):
        """Initialize or load trained models"""
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        model_path = "data/ai_models.pkl"
        
        if os.path.exists(model_path):
            print("📦 Loading pre-trained AI models...")
            self.models = joblib.load(model_path)
            self.scaler = joblib.load("data/scaler.pkl")
            self.label_encoder = joblib.load("data/encoder.pkl")
        else:
            print("🤖 Training new AI models...")
            self.train_models()
    
    def generate_synthetic_data(self, n_samples=5000):
        """Generate realistic synthetic financial data for training"""
        np.random.seed(42)
        
        sectors = ['FinTech', 'AgriTech', 'HealthTech', 'EdTech', 'CleanTech', 'SaaS', 'AutoTech', 'BioTech']
        
        data = {
            'valuation': np.random.lognormal(6.5, 1.2, n_samples),
            'revenue_growth': np.random.beta(2, 5, n_samples) * 100,
            'profit_margin': np.random.normal(15, 10, n_samples),
            'roi': np.random.exponential(30, n_samples) + np.random.normal(0, 15, n_samples),
            'market_share': np.random.beta(1.5, 8, n_samples) * 100,
            'competition_score': np.random.randint(1, 100, n_samples),
            'liquidity_score': np.random.beta(3, 2, n_samples) * 100,
            'sector_growth': np.random.normal(25, 15, n_samples),
            'company_age': np.random.exponential(5, n_samples) + 1,
            'funding_rounds': np.random.poisson(3, n_samples),
        }
        
        df = pd.DataFrame(data)
        df['sector'] = np.random.choice(sectors, n_samples)
        
        # Generate realistic exit strategies based on patterns
        def assign_strategy(row):
            if row['valuation'] > 3000 and row['revenue_growth'] > 40:
                return 'IPO'
            elif row['valuation'] < 1000 and row['competition_score'] > 70:
                return 'Strategic Sale'
            elif row['sector'] in ['HealthTech', 'FinTech'] and row['market_share'] > 30:
                return 'Merger'
            elif row['roi'] > 150:
                return 'Secondary Sale'
            elif row['liquidity_score'] < 40:
                return 'Buyback'
            else:
                return 'Private Equity Exit'
        
        df['exit_strategy'] = df.apply(assign_strategy, axis=1)
        
        # Add some noise for realism
        noise_idx = np.random.choice(n_samples, size=int(n_samples*0.1), replace=False)
        df.loc[noise_idx, 'exit_strategy'] = np.random.choice(df['exit_strategy'].unique(), len(noise_idx))
        
        return df
    
    def train_models(self):
        """Train machine learning models"""
        print("🔄 Training AI models...")
        
        df = self.generate_synthetic_data()
        
        # Prepare features
        X = df.drop(['exit_strategy', 'sector'], axis=1)
        y = df['exit_strategy']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_scaled, y_encoded)
        
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            random_state=42
        )
        gb_model.fit(X_scaled, y_encoded)
        
        # Clustering for market segmentation
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_scaled)
        
        # Save models
        self.models = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'kmeans': kmeans,
            'feature_names': X.columns.tolist()
        }
        
        # Save to disk
        os.makedirs("data", exist_ok=True)
        joblib.dump(self.models, "data/ai_models.pkl")
        joblib.dump(self.scaler, "data/scaler.pkl")
        joblib.dump(self.label_encoder, "data/encoder.pkl")
        
        print("✅ AI models trained and saved successfully!")
        print(f"📊 Model features: {len(X.columns)}")
        print(f"🎯 Strategies: {list(self.label_encoder.classes_)}")
    
    def predict_exit(self, input_data):
        """Predict exit strategy for a company"""
        try:
            # Prepare input features
            features = self.models['feature_names']
            
            # Create input DataFrame with all expected features
            input_dict = {
                'valuation': input_data.get('valuation', 1000),
                'revenue_growth': input_data.get('revenue_growth', 20),
                'profit_margin': input_data.get('profit_margin', 15),
                'roi': input_data.get('roi', 50),
                'market_share': input_data.get('market_share', 20),
                'competition_score': input_data.get('competition_score', 50),
                'liquidity_score': input_data.get('liquidity_score', 50),
                'sector_growth': input_data.get('sector_growth', 25),
                'company_age': input_data.get('company_age', 5),
                'funding_rounds': input_data.get('funding_rounds', 3)
            }
            
            input_df = pd.DataFrame([input_dict])[features]
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Get predictions from both models
            rf_pred = self.models['random_forest'].predict(input_scaled)[0]
            gb_pred = self.models['gradient_boosting'].predict(input_scaled)[0]
            
            # Get probabilities
            rf_proba = self.models['random_forest'].predict_proba(input_scaled)[0]
            gb_proba = self.models['gradient_boosting'].predict_proba(input_scaled)[0]
            
            # Ensemble prediction (weighted average)
            ensemble_proba = (rf_proba * 0.6 + gb_proba * 0.4)
            final_pred = np.argmax(ensemble_proba)
            
            # Decode prediction
            strategy = self.label_encoder.inverse_transform([final_pred])[0]
            
            # Get market cluster
            cluster = self.models['kmeans'].predict(input_scaled)[0]
            
            # Feature importance
            rf_importance = self.models['random_forest'].feature_importances_
            top_features = sorted(zip(features, rf_importance), 
                                key=lambda x: x[1], reverse=True)[:3]
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(input_data)
            
            # Determine optimal timing
            optimal_timing = self.predict_timing(input_data, strategy)
            
            # Get all strategy probabilities
            all_probabilities = {}
            for i, prob in enumerate(ensemble_proba):
                strategy_name = self.label_encoder.inverse_transform([i])[0]
                all_probabilities[strategy_name] = round(prob * 100, 1)
            
            return {
                'strategy': strategy,
                'confidence': round(ensemble_proba[final_pred] * 100, 1),
                'rf_confidence': round(rf_proba[rf_pred] * 100, 1),
                'gb_confidence': round(gb_proba[gb_pred] * 100, 1),
                'market_cluster': int(cluster),
                'top_factors': [(f[0], round(f[1]*100, 1)) for f in top_features],
                'all_probabilities': all_probabilities,
                'risk_score': risk_score,
                'optimal_timing': optimal_timing,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return default prediction
            return {
                'strategy': 'IPO',
                'confidence': 75.0,
                'risk_score': 50,
                'optimal_timing': '6-12 months',
                'error': str(e)
            }
    
    def calculate_risk_score(self, input_data):
        """Calculate risk score for a company"""
        risk_score = 20  # Base risk
        
        # Competition risk (0-25)
        competition = input_data.get('competition_score', 50)
        if competition > 80:
            risk_score += 25
        elif competition > 60:
            risk_score += 15
        elif competition > 40:
            risk_score += 5
        
        # Liquidity risk (0-20)
        liquidity = input_data.get('liquidity_score', 50)
        if liquidity < 30:
            risk_score += 20
        elif liquidity < 50:
            risk_score += 10
        
        # Growth risk (0-15)
        growth = input_data.get('revenue_growth', 20)
        if growth < 5:
            risk_score += 15
        elif growth < 10:
            risk_score += 8
        
        # Profitability risk (0-20)
        profit = input_data.get('profit_margin', 15)
        if profit < 0:
            risk_score += 20
        elif profit < 5:
            risk_score += 10
        
        # Market share risk (0-15)
        market_share = input_data.get('market_share', 20)
        if market_share < 5:
            risk_score += 15
        elif market_share < 10:
            risk_score += 7
        
        # Age risk (0-5)
        age = input_data.get('company_age', 5)
        if age < 2:
            risk_score += 5
        elif age < 4:
            risk_score += 2
        
        return min(risk_score, 95)  # Cap at 95
    
    def predict_timing(self, input_data, strategy):
        """Predict optimal exit timing"""
        base_timings = {
            'IPO': 18,
            'Strategic Sale': 6,
            'Merger': 12,
            'Buyback': 3,
            'Private Equity Exit': 24,
            'Secondary Sale': 9
        }
        
        # Adjust based on company metrics
        adjustments = 0
        
        # High ROI → faster exit
        if input_data.get('roi', 0) > 100:
            adjustments -= 4
        elif input_data.get('roi', 0) > 50:
            adjustments -= 2
        
        # Low liquidity → slower exit
        if input_data.get('liquidity_score', 50) < 40:
            adjustments += 4
        
        # High competition → faster exit
        if input_data.get('competition_score', 50) > 70:
            adjustments -= 3
        
        # High growth → slower exit (wait for more growth)
        if input_data.get('revenue_growth', 20) > 50:
            adjustments += 3
        
        # Calculate months
        base_months = base_timings.get(strategy, 12)
        months = max(1, base_months + adjustments)
        
        # Convert to human-readable format
        if months <= 3:
            return "Immediate (1-3 months)"
        elif months <= 6:
            return "Short-term (3-6 months)"
        elif months <= 12:
            return "Medium-term (6-12 months)"
        elif months <= 24:
            return f"Long-term ({months} months)"
        else:
            return "Extended timeline (24+ months)"
    
    def generate_insights(self, company_data, prediction):
        """Generate AI insights for the prediction"""
        name = company_data.get('name', 'This company')
        strategy = prediction['strategy']
        confidence = prediction['confidence']
        risk = prediction['risk_score']
        
        insights = []
        
        # Confidence insights
        if confidence > 90:
            insights.append(f"🤖 **High Confidence**: AI model shows strong certainty in {strategy} recommendation.")
        elif confidence > 80:
            insights.append(f"🤖 **Good Confidence**: AI model recommends {strategy} with reasonable certainty.")
        else:
            insights.append(f"🤖 **Moderate Confidence**: Consider additional factors for {strategy} decision.")
        
        # Risk insights
        if risk > 70:
            insights.append(f"⚠️ **Elevated Risk**: Risk score of {risk} suggests careful due diligence.")
        elif risk > 40:
            insights.append(f"⚖️ **Moderate Risk**: Risk score of {risk} indicates balanced opportunity.")
        else:
            insights.append(f"✅ **Low Risk**: Risk score of {risk} suggests stable investment profile.")
        
        # Strategy-specific insights
        if strategy == "IPO":
            insights.append(f"📈 **IPO Potential**: Strong valuation and growth metrics support public offering.")
        elif strategy == "Strategic Sale":
            insights.append(f"🤝 **Sale Opportunity**: Consider approaching strategic buyers in the {company_data.get('sector', '')} sector.")
        elif strategy == "Merger":
            insights.append(f"🔄 **Merger Potential**: Synergies with larger players could maximize value.")
        elif strategy == "Buyback":
            insights.append(f"💎 **Buyback Value**: Current valuation suggests share repurchase could benefit shareholders.")
        
        # Top factors from prediction
        if 'top_factors' in prediction and prediction['top_factors']:
            top_factor = prediction['top_factors'][0][0].replace('_', ' ').title()
            insights.append(f"🔍 **Key Driver**: {top_factor} was the most influential factor in this recommendation.")
        
        return insights

# Create global AI model instance
ai_model = ExitStrategyAI()