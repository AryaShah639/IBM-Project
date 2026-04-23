import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="✈️",
    layout="wide",
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_model()
model     = artifact["model"]
le_dict   = artifact["le_dict"]
features  = artifact["feature_names"]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("✈️ Customer Churn Prediction")
st.markdown(
    "This app predicts whether a **travel customer is likely to churn** "
    "using a trained **Random Forest** classifier."
)
st.divider()

# ── Layout: input | result ────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("🔢 Enter Customer Details")

    age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)

    frequent_flyer = st.selectbox(
        "Frequent Flyer",
        options=["No", "Yes", "No Record"],
    )

    annual_income = st.selectbox(
        "Annual Income Class",
        options=["Low Income", "Middle Income", "High Income"],
    )

    services_opted = st.slider(
        "Number of Services Opted", min_value=1, max_value=9, value=3, step=1
    )

    account_synced = st.radio(
        "Account Synced to Social Media?",
        options=["No", "Yes"],
        horizontal=True,
    )

    booked_hotel = st.radio(
        "Booked Hotel or Not?",
        options=["No", "Yes"],
        horizontal=True,
    )

    predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")

# ── Prediction logic ──────────────────────────────────────────────────────────
with right:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        # encode inputs
        ff_enc   = le_dict["FrequentFlyer"].transform([frequent_flyer])[0]
        ai_enc   = le_dict["AnnualIncomeClass"].transform([annual_income])[0]
        as_enc   = le_dict["AccountSyncedToSocialMedia"].transform([account_synced])[0]
        bh_enc   = le_dict["BookedHotelOrNot"].transform([booked_hotel])[0]

        input_df = pd.DataFrame(
            [[age, ff_enc, ai_enc, services_opted, as_enc, bh_enc]],
            columns=features,
        )

        pred        = model.predict(input_df)[0]
        proba       = model.predict_proba(input_df)[0]
        churn_prob  = proba[1] * 100
        stay_prob   = proba[0] * 100

        # result banner
        if pred == 1:
            st.error("⚠️ **This customer is likely to CHURN**")
        else:
            st.success("✅ **This customer is likely to STAY**")

        # probability gauge
        st.markdown("#### Churn Probability Breakdown")
        col1, col2 = st.columns(2)
        col1.metric("🔴 Churn Probability",  f"{churn_prob:.1f}%")
        col2.metric("🟢 Retention Probability", f"{stay_prob:.1f}%")

        # horizontal bar chart
        fig, ax = plt.subplots(figsize=(5, 1.4))
        ax.barh([""], [stay_prob],  color="#2ecc71", height=0.5, label="Stay")
        ax.barh([""], [churn_prob], color="#e74c3c", height=0.5,
                left=[stay_prob], label="Churn")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Stay vs Churn", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # feature importance mini-chart
        st.markdown("#### Feature Importance")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
        fi_df = fi_df.sort_values("Importance")

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        bars = ax2.barh(fi_df["Feature"], fi_df["Importance"], color="#3498db")
        ax2.set_xlabel("Importance Score")
        ax2.set_title("What Drives the Prediction?", fontsize=10)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    else:
        st.info("👈 Fill in the customer details on the left and click **Predict Churn**.")

        # show sample class distribution
        st.markdown("#### Dataset Class Distribution")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        labels  = ["Stayed (0)", "Churned (1)"]
        sizes   = [730, 224]
        colors  = ["#2ecc71", "#e74c3c"]
        ax3.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax3.set_title("Churn Distribution in Training Data")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Random Forest Classifier · Accuracy: ~87.4% · "
    "Dataset: Customer Travel Churn · B.Tech Gen AI Final Project"
)