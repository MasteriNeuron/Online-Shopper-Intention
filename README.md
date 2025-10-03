---
title: Online Shopper Intention
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🛍️ Online Shopper Intention Prediction

This project predicts whether an **online shopper intends to purchase** during their session.  
It uses a **Flask-based ML application** deployed on **Hugging Face Spaces**, with a frontend built using **HTML, CSS, and JavaScript**.

---

## 🚀 Demo
👉 Try it directly on Hugging Face Spaces (https://huggingface.co/spaces/Master89/RevenueRadar).

---

## 📂 Project Structure
```
📦 Online-Shopper-Intention  
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
│   ├── 📈 plots/  
│   └── 🎨 styles.css  
├── 🖼 templates/  
│   ├── 📊 eda.html  
│   ├── 🏠 home.html  
│   ├── 🔮 predict.html  
│   └── 🏋️ train.html  
├── 📘 Readme.md  
├── 🚀 app.py  
└── 📦 requirements.txt  

```
