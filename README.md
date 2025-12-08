# Loan-Repayment-Prediction-ML
Loan Approval Prediction uses machine learning to analyze factors like income and credit history, automating and improving the loan approval process. The project ensures accurate, efficient, and user-friendly predictions with a Gradio interface.


## 📖 Overview
Loan approval prediction is a critical task in the banking and finance industry. This project leverages machine learning to predict whether a loan application will be approved based on features such as applicant income, credit history, loan amount, and more. By automating the decision-making process, this system aims to reduce risks for financial institutions and improve customer satisfaction.

---

## 🚀 Features
- **Machine Learning Models**: Uses multiple algorithms like Random Forest, Logistic Regression, K-Nearest Neighbors, and Neural Networks.
- **Dynamic Output**: Displays prediction results with color-coded status (e.g., "Approved" in green, "Rejected" in red).
- **User-Friendly Interface**: Built with Gradio for real-time loan status prediction.
- **Feature Analysis**: Explores the impact of different features like credit history and income on loan approval.

---

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation and preprocessing
  - `scikit-learn`: Machine learning model development
  - `gradio`: Building interactive interfaces
  - `matplotlib`, `seaborn`: Data visualization

---

## 📊 Dataset
- **Description**: The dataset includes key features such as:
  - Applicant's Income
  - Coapplicant's Income
  - Loan Amount
  - Credit History
  - Property Area
- **Source**: (Mention if it’s from Kaggle, UCI, or custom data)
- **Data Preprocessing**:
  - Handling missing values
  - Encoding categorical variables
  - Balancing classes with SMOTE

---

## 🧠 Machine Learning Workflow
1. **Data Preprocessing**: Cleaning, scaling, and encoding data for optimal model performance.
2. **Model Training**:
   - Trains multiple algorithms (e.g., Logistic Regression, Random Forest) and evaluates them using metrics like accuracy, precision, recall, and ROC-AUC.
3. **Deployment**:
   - Uses Gradio for an interactive prediction interface.
4. **Dynamic Output**:
   - Displays predictions with color-coded text to indicate loan status.

---

## 📈 Evaluation Metrics
- **Accuracy**: Overall correctness of predictions.
- **Precision**: Reducing false positives (risky loans predicted as safe).
- **Recall**: Reducing false negatives (safe loans predicted as risky).
- **ROC-AUC**: Assessing model discrimination between approved and rejected loans.

---

## 🔍 Model Performance

## 🔍 Model Performance

The project evaluates multiple machine learning algorithms, including an Artificial Neural Network (ANN), to identify the best-performing model for loan approval prediction. Below are the performance metrics for each algorithm:

| **Model Name**            | **Accuracy (%)** | **Precision (%)** | **Confusion Matrix**        |
|---------------------------|------------------|-------------------|-----------------------------|
| Logistic Regression       | 74.56           | 77.03            | [[52, 34], [9, 74]]        |
| K-Nearest Neighbors       | 79.29           | 74.94            | [[56, 30], [14, 69]]       |
| Support Vector Machine    | 74.56           | 77.47            | [[51, 35], [8, 75]]        |
| Random Forest             | 82.84           | 85.53            | [[60, 26], [3, 80]]        |
| Decision Tree             | 79.88           | 79.09            | [[60, 26], [11, 72]]       |
| Artificial Neural Network | 77.78           | N/A              | N/A                        |

### Artificial Neural Network (ANN) Training Summary:
- **Training Process**:
  - Trained over 10 epochs using the Adam optimizer and binary cross-entropy loss function.
  - Showed gradual improvement in accuracy and reduction in loss during training and validation.
- **Performance Metrics**:
  - **Training Accuracy**: 77.06%
  - **Validation Accuracy**: 77.78%
  - **Test Accuracy**: 75.15%

### Key Observations:
- **Random Forest** achieves the highest accuracy (82.84%) and precision (85.53%), making it the most reliable model for this dataset.
- **K-Nearest Neighbors** and **Decision Tree** also perform well, with accuracy scores of 79.29% and 79.88%, respectively.
- **Artificial Neural Network** demonstrates competitive performance, showing the potential for further optimization with additional hyperparameter tuning or dataset expansion.



---
## 🖥️ Getting Started
### Prerequisites
- Python 3.7+
- Install required libraries:
  ```bash
  pip install -r requirements.txt


## 🌟Output Demo


![Image](https://github.com/user-attachments/assets/c65902bf-906f-4264-9770-aac3f992d379)


![Image](https://github.com/user-attachments/assets/816ddce9-8858-4826-872e-723a5d7605f2)



---

## 🤝 Contributing
    Contributions are welcome! Feel free to open issues or submit pull requests.

    
---

## 📜 License
This project is licensed under the MIT License.

