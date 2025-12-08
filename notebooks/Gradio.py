import numpy as np
import gradio as gr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Generate synthetic training data
# -----------------------------
np.random.seed(42)
X = np.random.rand(1000, 11) * [5000, 3000, 200, 360, 1, 1, 1, 3, 1, 1, 2]  # scale to feature ranges
y = np.random.randint(0, 2, 1000)  # binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
KNclassifier = KNeighborsClassifier(n_neighbors=5)
KNclassifier.fit(X_train, y_train)

# -----------------------------
# Prediction function
# -----------------------------
def predict_loan_status(applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                        credit_history, gender, married, dependents, education, self_employed, property_area):
    property_area_encoded = {"rural": 0, "semiurban": 1, "urban": 2}.get(property_area.lower(), -1)
    if property_area_encoded == -1:
        return "Invalid Property Area. Please enter 'Rural', 'Semiurban', or 'Urban'."

    user_input = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                            credit_history, gender, married, dependents, education,
                            self_employed, property_area_encoded]])
    user_input_scaled = scaler.transform(user_input)
    prediction = KNclassifier.predict(user_input_scaled)
    return "Loan Status: Paid" if prediction[0] == 1 else "Loan Status: Not Paid"

# -----------------------------
# Build the Gradio Blocks app
# -----------------------------
with gr.Blocks() as demo:  # white theme
    gr.Markdown("# Loan Prediction App")
    gr.Markdown("Enter the required details to predict loan status.")

    with gr.Row():
        applicant_income = gr.Number(label="Applicant's Income")
        coapplicant_income = gr.Number(label="Coapplicant's Income")
    
    with gr.Row():
        loan_amount = gr.Number(label="Loan Amount")
        loan_amount_term = gr.Number(label="Loan Amount Term (months)")
    
    with gr.Row():
        credit_history = gr.Radio([1, 0], label="Credit History (1 = Yes, 0 = No)")
        gender = gr.Radio([1, 0], label="Gender (1 = Male, 0 = Female)")
        married = gr.Radio([1, 0], label="Married (1 = Yes, 0 = No)")
    
    with gr.Row():
        dependents = gr.Number(label="Number of Dependents")
        education = gr.Radio([1, 0], label="Education (1 = Graduate, 0 = Not Graduate)")
        self_employed = gr.Radio([1, 0], label="Self Employed (1 = Yes, 0 = No)")
    
    property_area = gr.Radio(["Rural", "Semiurban", "Urban"], label="Property Area")
    output = gr.Textbox(label="Prediction")
    submit = gr.Button("Predict")

    # Connect button to function
    submit.click(
        fn=predict_loan_status,
        inputs=[applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                credit_history, gender, married, dependents, education,
                self_employed, property_area],
        outputs=output
    )

# Launch the app
demo.launch()
