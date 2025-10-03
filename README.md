---
title: Online Shopper Intention
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🛒 Online Shoppers’ Purchasing Intention Prediction
<img width="1914" height="834" alt="image" src="https://github.com/user-attachments/assets/e5e80834-016b-4c28-a0f1-7a1e134c76e4" />


This project, developed as part of a **PWSkills mini-hackathon**, predicts whether an online shopping session will result in a purchase using the **UCI Online Shoppers Purchasing Intention Dataset**.

With global e-commerce conversion rates averaging **1–3%**, accurately identifying high-intent sessions is critical for:

* ⚡ Boosting conversions
* 💰 Reducing cart abandonment
* 🎯 Enabling personalized marketing
* 🤝 Improving customer experience

The final system achieves an **F1-score of 0.92** and **ROC-AUC of 0.96**, and is deployed on **Hugging Face Spaces** as **RevenueRadar**.

---

## 📑 Introduction
<img width="1915" height="857" alt="image" src="https://github.com/user-attachments/assets/27f92028-2f8f-4fa3-ae55-8f44af1cbc90" />

E-commerce platforms struggle to convert browsing sessions into purchases. This project leverages **user behavior data (12,330 sessions, 18 features)** to build robust ML pipelines that:

* 📊 Preprocess data with advanced encoding, scaling, PCA, and clustering
* 🤖 Train and tune multiple models (XGBoost, LightGBM, RandomForest, GradientBoosting, cuML variants)
* 🎯 Optimize using **SMOTE** for imbalance and Bayesian hyperparameter search
* 📈 Deliver interpretable insights and actionable business recommendations

---

## 📂 Project Structure

```markdown
📦 Project
├── 📓 Notebooks/
│   ├── 🖼 images/
│   ├── 📊 EDA.ipynb
│   ├── 🤖 model_training.ipynb
│   └── 🧹 preprocessing.ipynb
├── 📊 datasets/
│   ├── 📂 processed/
│   │   ├── 🧹 clean.csv
│   │   ├── 🔑 kmeans.pkl
│   │   ├── 🔑 month_encoding_mapping.pkl
│   │   ├── 🔑 pca.pkl
│   │   ├── 🔑 power_transformer.pkl
│   │   ├── 🗂 preprocessed_data.csv
│   │   ├── 🔑 scaler.pkl
│   │   ├── 🔑 trained_model.pkl
│   │   └── 🔑 visitor_type_freq_mapping.pkl
│   └── 📂 raw/
│       └── 📑 online_shoppers_intention.csv
├── 📝 logs/
│   └── 📄 pipeline.log
├── 🤖 models/
│   ├── 📈 plots/
│   └── 🔑 best_model.pkl
├── 🛠 src/
│   ├── ⚙️ config/
│   │   ├── 📑 config.yaml
│   │   └── 📜 config.py
│   ├── 🔄 data_processing/
│   │   └── 🧾 preprocess.py
│   ├── 📝 logger/
│   │   └── 📜 logs.py
│   ├── 🔗 pipelines/
│   │   ├── 🔮 prediction_pipeline.py
│   │   └── 🏋️ training_pipeline.py
│   └── 🧩 utils/
│       ├── 📊 dash_app.py
│       ├── 📥 data_loader.py
│       └── 🛠 helper.py
├── 🎨 static/
│   └── 🎨 styles.css
├── 🖼 templates/
│   ├── 📊 eda.html
│   ├── 🏠 home.html
│   ├── 🔮 predict.html
│   └── 🏋️ train.html
├── 📘 Readme.md
├── 🚀 app.py
├── 🐳 Dockerfile
└── 📦 requirements.txt
```

---

## 📊 Dataset

* **Source:** [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset)
* **Size:** 12,330 rows × 18 columns → cleaned to 12,205 rows
* **Target:** `Revenue` (binary; 15.6% purchases, 84.4% non-purchases → imbalanced)
* **Key Features:**

  * Page visits & durations (Administrative, Informational, ProductRelated)
  * Engagement metrics (BounceRates, ExitRates, PageValues)
  * Contextual factors (Month, VisitorType, Weekend)

**Class imbalance:** handled with **SMOTE (5.4:1 → 1:1 balance)**

---

## 🔬 Methodology

### 🛠 Preprocessing

* Duplicate removal & robust imputation
* Encodings:

  * Target encoding → Month
  * Frequency encoding → VisitorType
* Scaling & Transformation: PowerTransformer + StandardScaler
* Dimensionality Reduction: PCA (95% variance → 15 components)
* Clustering: KMeans (k=3, silhouette optimized) added as feature
* Class balancing: SMOTE

### 📊 Feature Engineering

* Composite features: `Total_PageViews`, `Total_Duration`, `Avg_Time_Per_Page`, `Engagement_Score`
* Ratios: `Bounce_Exit_Ratio`
* Engagement_Score designed to capture business-relevant signals

### 🤖 Model Training & Optimization

* Models: RandomForest, GradientBoosting, LightGBM, XGBoost, cuML variants (if GPU)
* Hyperparameter Tuning: Bayesian Optimization (F1-focused)
* Best Model: **XGBoost**

---

## 📈 Results

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.96  |
| F1-Score  | 0.92  |
| Precision | 0.91  |
| Recall    | 0.93  |
| Log Loss  | 0.25  |

* ✅ Minimal overfitting
* ✅ Strong generalization
* ✅ Key driver: **Engagement_Score**

---

## 🌐 Web Application

Built with **Flask + Dash**, deployed as **RevenueRadar** on Hugging Face Spaces.

* `/` → Home
* `/train` → Train models
* `/predict` → Single/batch predictions
* `/eda` → Interactive dashboard (8+ Plotly charts)
* `/logs` → Logs & monitoring

---

## 🚀 Deployment

1. **Clone repo**

   ```bash
   git clone https://github.com/MasteriNeuron/Online-Shopper-Intention.git
   cd RevenueRadar
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run app**

   ```bash
   python app.py
   ```

4. Access at: `https://huggingface.co/spaces/Master89/RevenueRadar`

---

## 💡 Key Insights

* **New Visitors** convert more (24.9%) than Returning (14.1%)
* **Traffic Sources 1–3** drive ~70% of conversions
* **Engagement_Score** outperforms raw metrics in predicting purchases
* **Cluster 2 (high-engagement users)** shows the highest purchase probability

---

## 🔮 Future Scope

* 📊 SHAP-based feature explainability
* ⏱️ Real-time session analysis
* 📱 Mobile-first integration
* 🧠 Advanced deep learning models (e.g., LSTMs for sequential clickstreams)

---

## 🙏 Acknowledgment

Special Thanks to **Mr. Shubham Chaudhary** for his valuable contribution to this project.

---

## 📧 Contact

For questions, suggestions, or contributions:  
📩 [LinkedIn: Shubham Chaudhary](https://www.linkedin.com/in/shubham-chaudhary1802/)  

---

✨ *RevenueRadar helps e-commerce businesses identify high-intent users, personalize engagement, and boost revenue with data-driven insights.*

