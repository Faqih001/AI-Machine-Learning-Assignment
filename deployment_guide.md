# 🚀 Deployment Guide: AI for Climate Action

## Overview
This guide explains how to deploy the Carbon Emission Prediction model using Streamlit for web applications and as a standalone Python script.

## 📁 Project Structure
```
├── README.md                           # Project overview and documentation
├── carbon_emission_prediction.ipynb   # Main Jupyter notebook
├── carbon_emission_model.py           # Standalone Python script
├── streamlit_app.py                   # Streamlit web application
├── requirements.txt                   # General dependencies
├── streamlit_requirements.txt         # Streamlit-specific dependencies
├── data/                              # Dataset directory
├── models/                           # Trained models and components
├── results/                         # Visualizations and reports
└── deployment_guide.md              # This file
```

## 🎯 Deployment Options

### Option 1: Streamlit Web Application (Recommended)

#### Local Deployment
1. **Install Dependencies**
   ```bash
   pip install -r streamlit_requirements.txt
   ```

2. **Train the Model First** (if not already done)
   ```bash
   python carbon_emission_model.py
   ```

3. **Run Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the Application**
   - Open your browser and go to `http://localhost:8501`
   - The web interface will load with multiple tabs for prediction, analysis, and insights

#### Features of Streamlit App:
- 🔮 **Single Country Prediction**: Interactive form for predicting emissions
- 📊 **Batch Analysis**: Upload CSV files for multiple country analysis
- 📈 **Visualizations**: Model insights and performance metrics
- ℹ️ **About Section**: Project information and SDG alignment

#### Cloud Deployment (Streamlit Cloud)
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `streamlit_app.py` as the main file
   - Use `streamlit_requirements.txt` for dependencies

3. **Configuration**
   - App will automatically deploy and provide a public URL
   - Supports automatic updates from GitHub

### Option 2: Standalone Python Script

#### Running the Complete Pipeline
```bash
python carbon_emission_model.py
```

This will:
- Generate synthetic data (or load real data)
- Perform exploratory data analysis
- Train and compare multiple ML models
- Evaluate model performance
- Create visualizations
- Save trained models for deployment

#### Using the Trained Model
```python
from carbon_emission_model import CarbonEmissionPredictor

# Initialize predictor
predictor = CarbonEmissionPredictor()

# Make prediction for a country
country_data = {
    'GDP_per_capita': 25000,
    'Population': 50000000,
    'Energy_consumption_per_capita': 150,
    'Renewable_energy_pct': 30,
    'Industrial_production_index': 100,
    'Forest_area_pct': 35,
    'Urbanization_rate': 65,
    'Education_index': 0.75,
    'Healthcare_expenditure_pct': 6.0
}

prediction = predictor.predict_emissions(country_data)
recommendations = predictor.generate_policy_recommendations(prediction, country_data)
```

### Option 3: API Deployment (Flask/FastAPI)

#### Create API Endpoint
```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model components
with open('models/best_model_random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process data and make prediction
    # Return JSON response
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🔧 Configuration and Customization

### Environment Variables
```bash
# Set environment variables for production
export MODEL_PATH=./models/
export DATA_PATH=./data/
export RESULTS_PATH=./results/
```

### Custom Data Sources
To use real data instead of synthetic data:

1. **Replace data generation in `carbon_emission_model.py`**
   ```python
   # Instead of generate_synthetic_data()
   def load_real_data(self):
       # Load from World Bank API, UN databases, or CSV files
       self.data = pd.read_csv('path/to/real_data.csv')
   ```

2. **Update Streamlit app for real-time data**
   ```python
   # Add API integration for live data
   import requests
   
   def fetch_live_data(country_code):
       # Fetch from World Bank API or other sources
       pass
   ```

### Model Retraining
```bash
# Schedule regular retraining
crontab -e
# Add: 0 0 1 * * /path/to/python /path/to/carbon_emission_model.py
```

## 📊 Monitoring and Maintenance

### Model Performance Monitoring
- Track prediction accuracy over time
- Monitor for data drift
- Retrain model with new data periodically

### Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Error Handling
- Implement comprehensive error handling
- Add input validation
- Provide user-friendly error messages

## 🌍 Production Best Practices

### Security
- Validate all user inputs
- Implement rate limiting
- Use HTTPS in production
- Sanitize file uploads

### Performance
- Cache model predictions
- Optimize model loading
- Use CDN for static assets
- Implement request batching

### Scalability
- Use container deployment (Docker)
- Implement load balancing
- Consider serverless deployment
- Database caching for frequent requests

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r streamlit_requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Commands
```bash
# Build image
docker build -t carbon-emission-predictor .

# Run container
docker run -p 8501:8501 carbon-emission-predictor
```

## 📱 Mobile Optimization

The Streamlit app is responsive and works on mobile devices. For better mobile experience:
- Use Streamlit's mobile-friendly components
- Optimize visualizations for smaller screens
- Implement touch-friendly interfaces

## 🔗 Integration Examples

### Government Policy Platform
```python
# Integrate with policy management systems
def generate_policy_report(country_data):
    prediction = predict_emissions(country_data)
    recommendations = generate_recommendations(prediction, country_data)
    
    # Generate PDF report
    # Send to policy makers
    # Update policy database
```

### Corporate Sustainability Dashboard
```python
# Integrate with corporate reporting
def calculate_corporate_footprint(company_data):
    # Convert company data to country-level indicators
    # Make predictions
    # Generate sustainability report
```

## 🎓 Educational Use

### Classroom Integration
- Use for teaching ML concepts
- Demonstrate real-world AI applications
- Show SDG alignment in practice

### Research Applications
- Extend model for research projects
- Compare with other climate models
- Validate against real emission data

## 🚀 Advanced Deployment

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: carbon-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: carbon-predictor
  template:
    metadata:
      labels:
        app: carbon-predictor
    spec:
      containers:
      - name: carbon-predictor
        image: carbon-emission-predictor:latest
        ports:
        - containerPort: 8501
```

### AWS/Azure/GCP Deployment
- Use cloud ML services for model hosting
- Implement auto-scaling
- Set up monitoring and alerts
- Use managed databases for data storage

## 📞 Support and Troubleshooting

### Common Issues
1. **Model files not found**: Run training script first
2. **Memory errors**: Use smaller batch sizes
3. **Slow predictions**: Optimize model or use caching
4. **Visualization errors**: Check plotly/matplotlib versions

### Getting Help
- Check project documentation
- Review error logs
- Submit issues on GitHub
- Contact project maintainer

## 🎯 Next Steps

### Enhancements
- [ ] Real-time data integration
- [ ] Time series forecasting
- [ ] Advanced visualizations
- [ ] Multi-language support
- [ ] Advanced policy recommendations

### Production Readiness
- [ ] Security audit
- [ ] Performance testing
- [ ] User acceptance testing
- [ ] Documentation review
- [ ] Monitoring setup

---

**🌍 Deploy with confidence and contribute to climate action through AI!**
