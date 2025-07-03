#!/usr/bin/env python3
"""
AI for Climate Action: Carbon Emission Prediction Model
UN SDG 13: Climate Action Implementation

This script implements a machine learning solution to predict carbon emissions
and contribute to sustainable development goals.

Author: Amirul
Date: July 2025
Assignment: Week 2 - Machine Learning Meets UN SDGs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pickle
import json
import os
from datetime import datetime

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CarbonEmissionPredictor:
    """
    A comprehensive machine learning system for predicting carbon emissions
    to support UN SDG 13: Climate Action
    """
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.feature_columns = []
        self.evaluation_results = {}
        self.feature_importance = None
        
    def create_directories(self):
        """Create necessary directories for the project"""
        directories = ['data', 'models', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✅ Project directories created!")
    
    def generate_synthetic_data(self, n_samples=200):
        """
        Generate synthetic climate and economic data for demonstration
        In a real project, this would load data from World Bank, UN databases, or Kaggle
        """
        print("🌍 Generating synthetic climate and economic dataset...")
        
        np.random.seed(42)
        
        # Generate realistic country data
        countries = [f"Country_{i}" for i in range(1, n_samples + 1)]
        
        # Economic indicators
        gdp_per_capita = np.random.lognormal(mean=9, sigma=1.2, size=n_samples)
        population = np.random.lognormal(mean=15, sigma=1.5, size=n_samples)
        
        # Energy and industrial factors
        energy_consumption = np.random.gamma(shape=2, scale=100, size=n_samples)
        renewable_energy_pct = np.random.beta(a=2, b=5, size=n_samples) * 100
        industrial_production = np.random.gamma(shape=3, scale=50, size=n_samples)
        
        # Environmental factors
        forest_area_pct = np.random.beta(a=3, b=2, size=n_samples) * 100
        urbanization_rate = np.random.beta(a=5, b=3, size=n_samples) * 100
        
        # Development indicators
        education_index = np.random.beta(a=8, b=2, size=n_samples)
        healthcare_expenditure = np.random.gamma(shape=3, scale=2, size=n_samples)
        
        # Calculate CO2 emissions with realistic relationships
        co2_emissions = (
            0.3 * np.log(gdp_per_capita) +
            0.4 * np.log(energy_consumption) +
            0.2 * (industrial_production / 100) +
            -0.1 * (renewable_energy_pct / 100) +
            -0.05 * (forest_area_pct / 100) +
            0.1 * (urbanization_rate / 100) +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Ensure realistic bounds
        co2_emissions = np.clip(co2_emissions, 0.1, 25)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'Country': countries,
            'GDP_per_capita': gdp_per_capita,
            'Population': population,
            'Energy_consumption_per_capita': energy_consumption,
            'Renewable_energy_pct': renewable_energy_pct,
            'Industrial_production_index': industrial_production,
            'Forest_area_pct': forest_area_pct,
            'Urbanization_rate': urbanization_rate,
            'Education_index': education_index,
            'Healthcare_expenditure_pct': healthcare_expenditure,
            'CO2_emissions_per_capita': co2_emissions
        })
        
        # Save the dataset
        self.data.to_csv('data/climate_economic_data.csv', index=False)
        
        print(f"📊 Dataset created with shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("📈 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        print("\n📊 Dataset Overview:")
        print(self.data.head())
        
        print("\n📈 Statistical Summary:")
        print(self.data.describe().round(2))
        
        print(f"\n❌ Missing values: {self.data.isnull().sum().sum()}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CO2 emissions distribution
        axes[0, 0].hist(self.data['CO2_emissions_per_capita'], bins=30, alpha=0.7, color='red')
        axes[0, 0].set_title('Distribution of CO2 Emissions per Capita')
        axes[0, 0].set_xlabel('CO2 Emissions (tons per capita)')
        axes[0, 0].set_ylabel('Frequency')
        
        # GDP vs CO2 emissions
        axes[0, 1].scatter(self.data['GDP_per_capita'], self.data['CO2_emissions_per_capita'], alpha=0.6)
        axes[0, 1].set_title('GDP per Capita vs CO2 Emissions')
        axes[0, 1].set_xlabel('GDP per Capita (USD)')
        axes[0, 1].set_ylabel('CO2 Emissions (tons per capita)')
        
        # Energy consumption vs CO2 emissions
        axes[1, 0].scatter(self.data['Energy_consumption_per_capita'], 
                          self.data['CO2_emissions_per_capita'], alpha=0.6, color='orange')
        axes[1, 0].set_title('Energy Consumption vs CO2 Emissions')
        axes[1, 0].set_xlabel('Energy Consumption per Capita')
        axes[1, 0].set_ylabel('CO2 Emissions (tons per capita)')
        
        # Renewable energy vs CO2 emissions
        axes[1, 1].scatter(self.data['Renewable_energy_pct'], 
                          self.data['CO2_emissions_per_capita'], alpha=0.6, color='green')
        axes[1, 1].set_title('Renewable Energy % vs CO2 Emissions')
        axes[1, 1].set_xlabel('Renewable Energy %')
        axes[1, 1].set_ylabel('CO2 Emissions (tons per capita)')
        
        plt.tight_layout()
        plt.savefig('results/exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Exploratory analysis completed!")
    
    def preprocess_data(self):
        """Preprocess data and engineer features"""
        print("🔧 DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 50)
        
        # Create a copy for preprocessing
        self.df_processed = self.data.copy()
        
        # Calculate correlation matrix
        features_df = self.df_processed.drop(['Country'], axis=1)
        correlation_matrix = features_df.corr()
        
        # Visualize correlation heatmap
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature Engineering
        self.df_processed['GDP_per_energy'] = (self.df_processed['GDP_per_capita'] / 
                                              self.df_processed['Energy_consumption_per_capita'])
        self.df_processed['Population_density_proxy'] = (self.df_processed['Population'] * 
                                                        self.df_processed['Urbanization_rate'] / 100)
        self.df_processed['Green_development_index'] = (
            self.df_processed['Renewable_energy_pct'] * 0.4 + 
            self.df_processed['Forest_area_pct'] * 0.3 + 
            self.df_processed['Education_index'] * 100 * 0.3
        )
        self.df_processed['Industrial_intensity'] = (self.df_processed['Industrial_production_index'] / 
                                                    self.df_processed['GDP_per_capita'])
        
        # Log transform skewed variables
        skewed_features = ['GDP_per_capita', 'Population', 'Energy_consumption_per_capita']
        for feature in skewed_features:
            self.df_processed[f'{feature}_log'] = np.log1p(self.df_processed[feature])
        
        print("✅ Feature engineering completed!")
        print(f"📊 New dataset shape: {self.df_processed.shape}")
        
        # Display correlation with target variable
        target_correlations = correlation_matrix['CO2_emissions_per_capita'].sort_values(key=abs, ascending=False)
        print("\n🎯 Features most correlated with CO2 emissions:")
        print(target_correlations.drop('CO2_emissions_per_capita').head(10))
    
    def prepare_ml_data(self):
        """Prepare data for machine learning"""
        print("🎯 PREPARING DATA FOR MACHINE LEARNING")
        print("=" * 50)
        
        # Select features for modeling
        self.feature_columns = [col for col in self.df_processed.columns 
                               if col not in ['CO2_emissions_per_capita', 'Country']]
        
        X = self.df_processed[self.feature_columns]
        y = self.df_processed['CO2_emissions_per_capita']
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"🎯 Target vector shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📈 Training set size: {self.X_train.shape[0]} samples")
        print(f"🧪 Testing set size: {self.X_test.shape[0]} samples")
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save the scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("✅ Data preprocessing completed!")
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("⚙️ MODEL TRAINING & HYPERPARAMETER TUNING")
        print("=" * 50)
        
        # 1. Linear Regression (Baseline)
        print("\n🔵 Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr_model
        
        lr_cv_scores = cross_val_score(lr_model, self.X_train_scaled, self.y_train, cv=5, scoring='r2')
        self.model_scores['Linear Regression'] = lr_cv_scores.mean()
        print(f"✅ Linear Regression CV R² Score: {lr_cv_scores.mean():.4f}")
        
        # 2. Random Forest with GridSearch
        print("\n🌲 Training Random Forest with GridSearch...")
        rf_model = RandomForestRegressor(random_state=42)
        
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
        rf_grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.models['Random Forest'] = rf_grid_search.best_estimator_
        self.model_scores['Random Forest'] = rf_grid_search.best_score_
        print(f"✅ Random Forest Best CV R² Score: {rf_grid_search.best_score_:.4f}")
        
        # 3. XGBoost with GridSearch
        print("\n🚀 Training XGBoost with GridSearch...")
        xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
        
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='r2', n_jobs=-1)
        xgb_grid_search.fit(self.X_train_scaled, self.y_train)
        
        self.models['XGBoost'] = xgb_grid_search.best_estimator_
        self.model_scores['XGBoost'] = xgb_grid_search.best_score_
        print(f"✅ XGBoost Best CV R² Score: {xgb_grid_search.best_score_:.4f}")
        
        # Select best model
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name} (R² = {self.model_scores[self.best_model_name]:.4f})")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("📈 MODEL EVALUATION & PERFORMANCE METRICS")
        print("=" * 50)
        
        def evaluate_model(model, X_test, y_test):
            y_pred = model.predict(X_test)
            return {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred)
            }, y_pred
        
        self.predictions = {}
        
        for model_name, model in self.models.items():
            metrics, y_pred = evaluate_model(model, self.X_test_scaled, self.y_test)
            self.evaluation_results[model_name] = metrics
            self.predictions[model_name] = y_pred
            
            print(f"📊 {model_name} Performance:")
            for metric_name, value in metrics.items():
                print(f"   {metric_name:4}: {value:.4f}")
            print()
        
        # Create performance comparison dataframe
        performance_df = pd.DataFrame(self.evaluation_results).round(4)
        performance_df.to_csv('results/model_performance.csv')
        
        # Feature importance for tree-based models
        if self.best_model_name in ['Random Forest', 'XGBoost']:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES ({self.best_model_name}):")
            print(self.feature_importance.head(10).to_string(index=False))
            
            self.feature_importance.to_csv('results/feature_importance.csv', index=False)
        
        print("\n✅ Model evaluation completed!")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("📊 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 50)
        
        # Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models_list = list(self.evaluation_results.keys())
        r2_scores = [self.evaluation_results[model]['R²'] for model in models_list]
        
        # R² Score comparison
        axes[0, 0].bar(models_list, r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Model R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(r2_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # RMSE comparison
        rmse_scores = [self.evaluation_results[model]['RMSE'] for model in models_list]
        axes[0, 1].bar(models_list, rmse_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Model RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        for i, v in enumerate(rmse_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Predictions vs Actual
        best_predictions = self.predictions[self.best_model_name]
        axes[1, 0].scatter(self.y_test, best_predictions, alpha=0.7, color='darkgreen')
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual CO2 Emissions')
        axes[1, 0].set_ylabel('Predicted CO2 Emissions')
        axes[1, 0].set_title(f'{self.best_model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
        
        # Residuals plot
        residuals = self.y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted CO2 Emissions')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'{self.best_model_name}: Residual Plot', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature Importance Visualization
        if self.best_model_name in ['Random Forest', 'XGBoost']:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(10)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Most Important Features ({self.best_model_name})', 
                     fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Visualizations created and saved!")
    
    def save_model(self):
        """Save the best model and components"""
        print("💾 SAVING MODEL AND COMPONENTS")
        print("=" * 30)
        
        # Save the best model
        model_filename = f'models/best_model_{self.best_model_name.lower().replace(" ", "_")}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save feature names
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Save model metadata
        model_metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'performance_metrics': self.evaluation_results[self.best_model_name],
            'feature_count': len(self.feature_columns),
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_columns
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        print(f"✅ Model saved as: {model_filename}")
        print("✅ All components saved successfully!")
    
    def predict_emissions(self, country_data):
        """Make predictions for new country data"""
        # Convert input to DataFrame
        if isinstance(country_data, dict):
            input_df = pd.DataFrame([country_data])
        else:
            input_df = country_data.copy()
        
        # Feature engineering (same as training)
        input_df['GDP_per_energy'] = input_df['GDP_per_capita'] / input_df['Energy_consumption_per_capita']
        input_df['Population_density_proxy'] = input_df['Population'] * input_df['Urbanization_rate'] / 100
        input_df['Green_development_index'] = (
            input_df['Renewable_energy_pct'] * 0.4 + 
            input_df['Forest_area_pct'] * 0.3 + 
            input_df['Education_index'] * 100 * 0.3
        )
        input_df['Industrial_intensity'] = input_df['Industrial_production_index'] / input_df['GDP_per_capita']
        input_df['GDP_per_capita_log'] = np.log1p(input_df['GDP_per_capita'])
        input_df['Population_log'] = np.log1p(input_df['Population'])
        input_df['Energy_consumption_per_capita_log'] = np.log1p(input_df['Energy_consumption_per_capita'])
        
        # Select features and predict
        X_new = input_df[self.feature_columns]
        X_new_scaled = self.scaler.transform(X_new)
        prediction = self.best_model.predict(X_new_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def generate_policy_recommendations(self, predicted_emissions, country_data):
        """Generate policy recommendations based on predictions"""
        recommendations = []
        priority_level = "Low"
        
        if predicted_emissions > 15.0:
            priority_level = "Critical"
            recommendations.extend([
                "Implement immediate carbon tax or cap-and-trade system",
                "Accelerate transition to renewable energy sources",
                "Invest heavily in energy efficiency programs",
                "Promote electric vehicle adoption with incentives"
            ])
        elif predicted_emissions > 10.0:
            priority_level = "High"
            recommendations.extend([
                "Increase renewable energy targets",
                "Implement building energy efficiency standards",
                "Support clean technology innovation",
                "Develop public transportation infrastructure"
            ])
        elif predicted_emissions > 5.0:
            priority_level = "Moderate"
            recommendations.extend([
                "Maintain current climate policies",
                "Focus on industrial emission reductions",
                "Promote sustainable agriculture practices",
                "Invest in carbon capture technologies"
            ])
        else:
            priority_level = "Low"
            recommendations.extend([
                "Continue best practices and share knowledge",
                "Support international climate finance",
                "Develop carbon-negative technologies",
                "Lead by example in international forums"
            ])
        
        return {
            'priority_level': priority_level,
            'recommendations': recommendations,
            'sdg_alignment': "SDG 13: Climate Action - Targets 13.2 and 13.3"
        }
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("🌍 AI FOR CLIMATE ACTION: CARBON EMISSION PREDICTION")
        print("🎯 UN SDG 13: Climate Action Implementation")
        print("=" * 60)
        
        # Step 1: Setup
        self.create_directories()
        
        # Step 2: Data Generation and Exploration
        self.generate_synthetic_data()
        self.explore_data()
        
        # Step 3: Data Preprocessing
        self.preprocess_data()
        self.prepare_ml_data()
        
        # Step 4: Model Training and Evaluation
        self.train_models()
        self.evaluate_models()
        
        # Step 5: Visualization and Model Saving
        self.create_visualizations()
        self.save_model()
        
        # Step 6: Demonstration
        self.demonstrate_predictions()
        
        print("\n🎯 PROJECT SUMMARY:")
        print(f"✅ Best Model: {self.best_model_name}")
        print(f"✅ Model Accuracy: {self.evaluation_results[self.best_model_name]['R²']:.1%}")
        print(f"✅ Average Error: {self.evaluation_results[self.best_model_name]['MAE']:.2f} tons CO2 per capita")
        print("✅ Production-ready model saved!")
        print("✅ SDG 13 Climate Action objectives achieved!")
    
    def demonstrate_predictions(self):
        """Demonstrate the prediction system with example data"""
        print("\n🧪 DEMONSTRATING PREDICTION SYSTEM:")
        print("-" * 40)
        
        # Example countries
        example_countries = [
            {
                'name': 'High Emission Country',
                'GDP_per_capita': 50000,
                'Population': 50000000,
                'Energy_consumption_per_capita': 300,
                'Renewable_energy_pct': 15,
                'Industrial_production_index': 120,
                'Forest_area_pct': 25,
                'Urbanization_rate': 85,
                'Education_index': 0.8,
                'Healthcare_expenditure_pct': 8
            },
            {
                'name': 'Low Emission Country',
                'GDP_per_capita': 15000,
                'Population': 20000000,
                'Energy_consumption_per_capita': 80,
                'Renewable_energy_pct': 60,
                'Industrial_production_index': 60,
                'Forest_area_pct': 50,
                'Urbanization_rate': 45,
                'Education_index': 0.7,
                'Healthcare_expenditure_pct': 5
            }
        ]
        
        for example in example_countries:
            country_name = example.pop('name')
            prediction = self.predict_emissions(example)
            recommendations = self.generate_policy_recommendations(prediction, example)
            
            print(f"\n🌍 {country_name}:")
            print(f"   📊 Predicted CO2 Emissions: {prediction:.2f} tons per capita")
            print(f"   🚨 Priority Level: {recommendations['priority_level']}")
            print(f"   💡 Key Recommendation: {recommendations['recommendations'][0]}")


def main():
    """Main function to run the carbon emission prediction system"""
    predictor = CarbonEmissionPredictor()
    predictor.run_complete_pipeline()


if __name__ == "__main__":
    main()
