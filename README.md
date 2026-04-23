# IBM-Project

# ✈️ Customer Churn Prediction using Random Forest

> **B.Tech – Gen AI | 2nd Semester | Individual Final Project**

---

## 📌 Project Overview

This project builds an end-to-end Machine Learning solution to predict whether a travel customer will **churn (leave)** or **stay**, using a **Random Forest Classifier**. It covers the full ML pipeline — from data preprocessing and model training to deployment as a live web application using **Streamlit Cloud**.

---

## 🗂️ Repository Structure

```
customer-churn-app/
│
├── app.py               # Streamlit web application
├── model.pkl            # Trained Random Forest model + encoders
├── requirements.txt     # Python dependencies
├── Customertravel.csv   # Dataset used for training
└── README.md            # Project documentation
```

---

## 📊 Dataset Description

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Age of the customer |
| `FrequentFlyer` | Categorical | Yes / No / No Record |
| `AnnualIncomeClass` | Categorical | Low / Middle / High Income |
| `ServicesOpted` | Numerical | Number of services opted (1–9) |
| `AccountSyncedToSocialMedia` | Categorical | Yes / No |
| `BookedHotelOrNot` | Categorical | Yes / No |
| `Target` | Binary | **0 = Stay**, **1 = Churn** |

- **Total Records:** 954
- **Churn Rate:** ~23.5% (224 churned, 730 stayed)

---

## ⚙️ ML Workflow

1. **Data Loading & Exploration** — shape, types, missing values
2. **Preprocessing** — Label Encoding of categorical features
3. **Train/Test Split** — 80% train, 20% test
4. **Model Training** — `RandomForestClassifier(n_estimators=100)`
5. **Evaluation** — Accuracy, Confusion Matrix, ROC Curve, Feature Importance

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~87.4% |
| **Algorithm** | Random Forest |
| **Test Size** | 20% |

### 🔑 Top Features Driving Churn

| Feature | Importance |
|---|---|
| Age | 29.9% |
| Services Opted | 23.5% |
| Frequent Flyer | 18.1% |
| Annual Income Class | 13.7% |
| Account Synced to Social Media | 9.9% |
| Booked Hotel or Not | 4.9% |

---

## 🚀 Streamlit App

The trained model is deployed as an interactive web app where users can enter customer details and instantly get a churn prediction.

### Features of the App
- 🎚️ Input sliders and dropdowns for all 6 features
- 🔴🟢 Churn vs Retention probability display
- 📊 Stay vs Churn probability bar chart
- 📈 Feature importance visualization
- 🥧 Dataset class distribution pie chart

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-churn-app.git
cd customer-churn-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Cloud)

1. Push all files to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Deploy an app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — your app goes live at:

```
https://your-app-name.streamlit.app
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
scikit-plot
matplotlib
```

---

## 📚 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training and evaluation |
| `matplotlib` | Visualizations |
| `streamlit` | Web app deployment |
| `pickle` | Model serialization |

---

## 👤 Author

- **Course:** B.Tech – Gen AI (2nd Semester)
- **Project Type:** Individual Final Project
- **Institution:** *(Add your college name)*
- **Student Name:** *(Add your name)*
- **Enrollment No.:** *(Add your enrollment number)*

---

## 📄 License

This project is submitted as part of academic coursework and is intended for educational purposes only.
