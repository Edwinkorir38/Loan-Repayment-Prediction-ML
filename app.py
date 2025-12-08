import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------------
# Generate synthetic training data
# -----------------------------
np.random.seed(42)
X = np.random.rand(1000, 11) * [5000, 3000, 200, 360, 1, 1, 1, 3, 1, 1, 2]
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

KNclassifier = KNeighborsClassifier(n_neighbors=5)
KNclassifier.fit(X_train, y_train)

# -----------------------------
# Prediction function
# -----------------------------
def predict_loan_status(inputs):
    applicant_income, coapplicant_income, loan_amount, loan_amount_term, \
    credit_history, gender, married, dependents, education, self_employed, property_area = inputs

    property_area_encoded = {"rural": 0, "semiurban": 1, "urban": 2}.get(property_area.lower(), -1)
    if property_area_encoded == -1:
        return "Invalid Property Area. Please enter 'Rural', 'Semiurban', or 'Urban'.", "error", None

    user_input = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                            credit_history, gender, married, dependents, education,
                            self_employed, property_area_encoded]])
    user_input_scaled = scaler.transform(user_input)
    
    prediction = KNclassifier.predict(user_input_scaled)[0]
    probability = KNclassifier.predict_proba(user_input_scaled)[0]  # [Not Paid, Paid]

    if prediction == 1:
        return "✅ Loan Status: Paid", "success", probability
    else:
        return "❌ Loan Status: Not Paid", "error", probability

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.set_page_config(page_title="Loan Prediction App", page_icon="💰", layout="wide")
st.title("💰 Loan Prediction App")
st.markdown("Interactive real-time loan prediction. Update any field to see the prediction instantly.")

# -----------------------------
# Two-column interactive inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant's Income", step=1000.0, format="%.2f", key="ai")
    coapplicant_income = st.number_input("Coapplicant's Income", step=1000.0, format="%.2f", key="ci")
    loan_amount = st.number_input("Loan Amount", step=1000.0, format="%.2f", key="la")
    loan_amount_term = st.number_input("Loan Amount Term (months)", step=1, key="lat")
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1, key="dep")

with col2:
    credit_history = st.selectbox("Credit History", [1, 0], key="ch")
    gender = st.selectbox("Gender", [1, 0], key="g")
    married = st.selectbox("Married", [1, 0], key="m")
    education = st.selectbox("Education", [1, 0], key="e")
    self_employed = st.selectbox("Self Employed", [1, 0], key="se")
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"], key="pa")

# -----------------------------
# Real-time prediction
# -----------------------------
inputs = [applicant_income, coapplicant_income, loan_amount, loan_amount_term,
          credit_history, gender, married, dependents, education, self_employed, property_area]

result_text, result_type, probability = predict_loan_status(inputs)

# Display status
if result_type == "success":
    st.success(result_text)
else:
    st.error(result_text)

# Display probability chart
if probability is not None:
    st.subheader("Prediction Probability")
    labels = ["Not Paid", "Paid"]
    colors = ["#FF4B4B", "#4CAF50"]
    
    fig, ax = plt.subplots(figsize=(4,2))
    bars = ax.bar(labels, probability, color=colors, alpha=0.8)
    ax.set_ylim([0, 1])
    
    # Annotate percentages
    for bar, prob in zip(bars, probability):
        ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02, f"{prob*100:.1f}%", 
                ha='center', fontweight='bold')
    
    ax.set_ylabel("Probability")
    ax.set_title("Loan Repayment Probability")
    st.pyplot(fig)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and KNeighborsClassifier")

