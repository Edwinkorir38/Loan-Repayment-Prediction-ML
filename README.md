
# 💰 Loan Repayment Prediction (Machine Learning)

Loan Approval Prediction uses machine learning to analyze factors like income, credit history, and property area — automating and improving the loan approval process.  
This project provides **accurate, efficient, and user-friendly loan prediction** using a **Gradio interface**.

---

## 📌 Status & Tech Stack
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-ML%20Interface-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-success)
![GitHub repo size](https://img.shields.io/github/repo-size/Edwinkorir38/loan-repayment-prediction-ml)
![GitHub stars](https://img.shields.io/github/stars/Edwinkorir38/loan-repayment-prediction-ml?style=social)
---

## 📖 Overview  
Loan approval prediction is a crucial task in the finance world.  
This system uses **machine learning models** to predict whether a loan should be approved based on 11 borrower-related features.

It helps financial institutions:

- Reduce risk  
- Make faster decisions  
- Improve customer experience  

---

## 🚀 Features
✔️ Multiple ML Models (LR, RF, KNN, SVM, ANN)  
✔️ Real-time prediction with Gradio UI  
✔️ Clean, beginner-friendly interface  
✔️ Color-coded prediction output  
✔️ Full ML pipeline: preprocessing → modeling → evaluation  

---

## 🛠️ Technologies Used
- **Python**
- **NumPy**, **pandas**
- **Scikit-Learn**
- **TensorFlow/Keras** (for ANN)
- **Matplotlib**, **Seaborn**
- **Gradio** (for deployment)

---

## 📊 Dataset
The dataset includes:

- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Amount Term  
- Credit History  
- Property Area  
- Employment status  
- Education level  
- Number of dependents  

  

### Data Processing Includes:
- Handling missing values  
- Encoding categorical variables  
- Scaling numerical values  
- Class balancing using **SMOTE**  

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Preprocessing  
- Handling missing values  
- Scaling and encoding  
- SMOTE oversampling  

### 2️⃣ Model Training  
Trained and evaluated:

- Logistic Regression  
- KNN  
- Support Vector Machine  
- Random Forest  
- Decision Tree  
- Artificial Neural Network  

### 3️⃣ Deployment  
The best model is deployed through **Gradio** and **streamlit**

### 4️⃣ Dynamic Output  
- Green = Approved  
- Red = Rejected  

---

## 📈 Model Performance Comparison

| **Model**                | **Accuracy (%)** | **Precision (%)** | **Confusion Matrix**       |
|--------------------------|------------------|-------------------|-----------------------------|
| Logistic Regression      | 74.56            | 77.03            | [[52, 34], [9, 74]]        |
| K-Nearest Neighbors      | 79.29            | 74.94            | [[56, 30], [14, 69]]       |
| Support Vector Machine   | 74.56            | 77.47            | [[51, 35], [8, 75]]        |
| Random Forest            | **82.84**        | **85.53**        | [[60, 26], [3, 80]]        |
| Decision Tree            | 79.88            | 79.09            | [[60, 26], [11, 72]]       |
| Artificial Neural Network| 77.78            | N/A              | N/A                        |

### 🔬 ANN Summary
- 10 epochs  
- Optimizer: **Adam**  
- Loss: **Binary Crossentropy**  
- Validation Accuracy: **77.78%**  
- Test Accuracy: **75.15%**

👉 **Random Forest is the top-performing model.**

---

## 🖥️ Getting Started

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/your-username/Loan-Repayment-Prediction-ML.git
cd Loan-Repayment-Prediction-ML
```
### ✅ 2. Install Dependencies
```
bash
pip install -r requirements.txt
```

### ✅ 3. Run the Gradio App
```
bash
python app.py
```
### 🌟 Output Demo
#### 🔍 Prediction Interface

![alt text](Images/image.png)

#### 📊 Probability & Analysis
![alt text](Images/image-1.png)
### 🤝 Contributing

Contributions are welcome ,  feel free to open issues or submit pull requests.

### 📜 License

Licensed under the MIT License.
## Project Structure
```
.
├── Data/
│   └── loan_dataset.csv
│
├── Images/
│   ├── image-1.png
│   └── image.png
│
├── notebooks/
│   ├── Gradio.py
│   └── loan_repayment.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE

```
## 🌐 Live Demo
Try the deployed machine learning application here:

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://loan-repayment-prediction-ml-f9em84ghnnxntz2yrvvcf4.streamlit.app/)
