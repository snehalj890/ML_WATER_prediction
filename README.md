# 💧 Water Quality Classification and Prediction System

A production-grade Machine Learning web application built to evaluate drinking water potability based on key chemical and physical water quality parameters. Developed as part of **Mission P** for career portfolio presentation.

---

## 🚀 Live Demo & Repository
* **GitHub Repository:** [snehalj890/water-quality-ml-system_classification_](https://github.com/snehalj890/water-quality-ml-system_classification_)

---

## 📋 Project Overview
Access to safe drinking water is critical to public health. This project leverages supervised machine learning classification algorithms to analyze water sample parameters and predict whether a given sample is **Potable (Safe)** or **Not Potable (Unsafe)** for human consumption. 

The application features a modern, immersive **Streamlit** user interface with dark glassmorphic design principles and responsive input controls for real-time inference.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 
* **Frontend / UI:** Streamlit (Custom CSS ocean theme styling)
* **Machine Learning & Modeling:** Scikit-Learn, Pandas, NumPy, Joblib (`water_quality_model.pkl`)
* **Version Control:** Git, GitHub Desktop
* **Deployment Platform:** Streamlit Community Cloud

---

## 📊 Dataset & Input Parameters
The model evaluates water quality using the following 9 physicochemical parameters:
1. **pH:** Measures the acid-base balance of water (Optimal: 6.5 – 8.5).
2. **Hardness:** Capacity of water to precipitate soap (Calcium and Magnesium content).
3. **Solids (Total Dissolved Solids - TDS):** Amount of minerals, salts, or metals dissolved in water.
4. **Chloramines:** Amount of chloramines added as disinfectants.
5. **Sulfate:** Dissolved minerals originating from rocks and soils.
6. **Conductivity:** Ability of water to conduct an electrical current.
7. **Organic Carbon:** Amount of organic carbon present in the water.
8. **Trihalomethanes:** Chemical compounds formed when chlorine interacts with organic matter.
9. **Turbidity:** Measure of the cloudiness or haziness of water.

---

## 📂 Project Directory Structure
```text
water-quality-ml-system_classification_/
│
├── app.py                     # Main Streamlit web application script
├── water_quality_model.pkl    # Trained machine learning classification model
├── requirements.txt           # Python package dependencies
├── runtime.txt                # Python runtime version configuration
└── README.md                  # Project documentation
