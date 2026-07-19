```markdown
# 🐟 Pelagic Fish Age Predictor

> A machine learning web application for predicting pelagic fish age based on environmental conditions and physical traits.

## 🎯 What This Project Demonstrates

This project showcases my machine learning and software development skills through:

- **ML Modeling**: Random Forest Regressor with hyperparameter optimization (GridSearchCV)
- **Data Preprocessing**: Ordinal encoding for categorical variables
- **Model Evaluation**: Comprehensive error analysis with visualizations
- **Web Deployment**: Interactive Streamlit application with professional UI
- **Code Organization**: Clean project structure with modular design

## 🚀 Quick Start (For Employers)

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/lee-ust/pelagic-fish-age-predictor.git
cd pelagic-fish-age-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web App
```bash
cd scripts
streamlit run front_end.py
```

### 4. Open in Browser
The app will open at `http://localhost:8501` - start predicting fish ages instantly!

### One-Click Launch (Alternative)
```bash
python run.py
```

## 📊 Model Performance Snapshot

| Metric | Value |
|--------|-------|
| **RMSE** | 2.90 years |
| **MAE** | 1.98 years |
| **R² Score** | 0.744 |

*View full performance analysis and visualizations in the web app's "Model Performance" page.*

## 🧠 Technical Highlights

- **Algorithm**: Random Forest Regressor
- **Features**: 9 features (2 categorical, 6 numerical, 1 habitat)
- **Optimization**: GridSearchCV with 3-fold cross-validation
- **Preprocessing**: OrdinalEncoder for categorical variables

### Features Used

| Feature | Type | Range |
|---------|------|-------|
| Common_Name | Categorical | 8 species |
| Habitat_Type | Categorical | Marine |
| Obs_Depth_m | Numerical | 0 - 1000 m |
| Body_Length_cm | Numerical | 78.3 - 739.2 cm |
| Body_Weight_kg | Numerical | 8.97 - 1270.96 kg |
| Sex | Categorical | juvenile/male/female |
| Water_Temp_C | Numerical | 1 - 30°C |
| Salinity_ppt | Numerical | 33 - 37 ppt |
| pH | Numerical | 7.5 - 8.4 |

## 📁 Project Structure

```
pelagic-fish-age-predictor/
├── README.md                       # Documentation
├── requirements.txt                # Python dependencies
├── run.py                          # Cross-platform Python launcher
├── run_app.bat                     # ⭐ Windows batch file (double-click to run)
├── .gitignore                      # Version control exclusions
├── .gitattributes                  # Git LFS configuration
├── data/
│   ├── pelagic_training_set.csv    # Training data (223 KB)
│   └── pelagic_test_set.csv        # Test data (519 KB)
├── output/
│   ├── model_error.png             # Prediction error visualization
│   ├── pelagic_fish_plot.png       # GGPairs distribution & Pearson R
│   ├── pelagic_mi.png              # Mutual Information visualization
│   ├── pelagic_nmi.png             # Normalized Mutual Information
│   └── pelagic_spearman_r.png      # Spearman correlation visualization
└── scripts/
    ├── front_end.py                # ⭐ Main Streamlit web app
    ├── fish_age_model.pkl          # ⭐ Trained Random Forest model
    ├── model.ipynb                 # Model training & evaluation
    ├── create_csv.py               # Training/test set creation
    └── data_analysis.r             # Exploratory data analysis
```

## 🔬 Try It Out!

The web app allows you to:

1. **Input Fish Characteristics**: Use the intuitive interface to enter environmental parameters and physical traits
2. **Handle Missing Data**: Mark up to 3 features as "unknown" with accuracy impact warnings
3. **Get Instant Predictions**: Receive age estimates with confidence context
4. **View Model Performance**: Explore error analysis and visualizations

### Making a Prediction

1. Enter environmental parameters (depth, temperature, salinity, pH)
2. Input physical traits (length, weight)
3. Select species and sex
4. Optionally mark up to 3 features as "unknown"
5. Click **Predict Age** to get the estimated fish age

## 📊 Data Analysis & Visualizations

The `output/` directory contains several visualizations from the exploratory data analysis:

| File | Description |
|------|-------------|
| `pelagic_fish_plot.png` | GGPairs showing distributions and Pearson correlation between all features |
| `pelagic_mi.png` | Mutual Information between features and target variable |
| `pelagic_nmi.png` | Normalized Mutual Information |
| `pelagic_spearman_r.png` | Spearman rank correlation matrix |
| `model_error.png` | Model prediction errors by age group |

### Key Findings from Data Analysis

- **Body Length** and **Body Weight** are the most important predictors of fish age
- **Water Temperature** shows moderate correlation with age
- The model performs best for middle-aged fish (2-8 years)
- Prediction errors are highest for extreme age groups

## 🛠️ Development Tools

The full development pipeline includes:

| Tool/Language | Purpose |
|---------------|---------|
| **Python** | Model training, preprocessing, web application |
| **R** | Exploratory data analysis and visualizations |
| **Streamlit** | Interactive web interface |
| **Scikit-learn** | ML modeling and evaluation |
| **Pandas** | Data manipulation |
| **Matplotlib/Seaborn** | Data visualization |

## 🔧 Installation Details

### Dependencies

```
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
joblib==1.3.2
Pillow==10.1.0
```

### R Dependencies (For Data Analysis)

If you want to run the exploratory data analysis:

```r
install.packages(c("tidyverse", "ggplot2", "GGally", "ggsci","Hmisc","ggcorrplot","praznik"))
```

## 📈 Model Development Process

### 1. Data Preparation
- Ordinal encoding for categorical variables
- Train/Test split: 70/30
- No scaling applied (tree-based model)

### 2. Model Training
- **Algorithm**: Random Forest Regressor
- **Cross-validation**: 10-fold with RMSE scoring
- **Hyperparameter Optimization**: GridSearchCV with 54 combinations

### 3. Model Evaluation
- RMSE: 2.90 years
- MAE: 1.98 years
- R²: 0.744
- Performance by age group analysis

## 💡 Understanding the Predictions

The model's predictions come with context:

- **Confidence**: Model RMSE is 2.90 years
- **Best Performance**: Ages 2-8 years
- **Limitations**: Higher errors for very young (0-2) and old (8-10+) fish

## 📝 Notes on Model Limitations

- **Age Range**: Trained on ages 0-33.7 years; extrapolation may be unreliable
- **Species Coverage**: Limited to 8 pelagic species
- **Data Distribution**: Model may perform differently for underrepresented groups
- **Missing Data**: Up to 3 features can be marked unknown, but accuracy will decrease

## 🚧 Future Improvements

- [ ] Add confidence intervals for predictions
- [ ] Support for additional species
- [ ] Real-time data input via API
- [ ] Mobile-responsive design
- [ ] Model retraining pipeline

## 👨‍💻 About the Developer

This project demonstrates my ability to:

- Build and deploy machine learning models
- Create interactive web applications
- Handle real-world data challenges
- Present complex results clearly
- Write production-ready code

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- Streamlit for web application framework
- Fisheries data provided by [Source](https://www.kaggle.com/datasets/maulikgajera/aquatic-wildlife-atlas-global-species-records)

---

## 📞 Contact

**Your Name**
- GitHub: [@lee-ust](https://github.com/lee-ust)
- Email: yleebi@connect.ust.hk

---

**Made with ❤️ for sustainable fisheries management**

## ⚡ Quick Commands Cheat Sheet

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
cd scripts && streamlit run front_end.py

# Or use the launcher
python run.py

# Check model performance
python -c "import joblib; print(joblib.load('scripts/fish_age_model.pkl'))"
```

## 🐛 Troubleshooting

### Issue: Module not found
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

### Issue: Model file not found
```bash
# Verify model exists
ls scripts/fish_age_model.pkl
```

### Issue: Port already in use
```bash
# Use a different port
streamlit run scripts/front_end.py --server.port 8502
```

### Issue: Missing visualization images
```bash
# Images are in the output/ directory
ls output/*.png
```

---

**⬆️ Back to Top** | **📧 Contact** | **📄 License**
```
