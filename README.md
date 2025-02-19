# -MACHINE-LEARNING-MODEL-IMPLEMENTATION
**Name:** Shriraj Dharmadhikari  
**Company:** CODTECH IT SOLUTIONS  
**ID:** CT08PEP  
**Domain:** Python Programming  
**Duration:** 25th Jan to 25th Feb  
**Mentor:** Neela Santosh  

---
# **Iris Flower Classification Using Random Forest**

## **Overview**
- This project uses the **Iris dataset** to train a **Random Forest Classifier** for species classification.
- It involves **data preprocessing, visualization, model training, and performance evaluation**.

---

## **Objective**
- Classify iris flowers into three species (**setosa, versicolor, virginica**).
- Train and evaluate a **Random Forest model** to improve accuracy.

---

## **Key Activities**
### **1. Data Loading and Preprocessing**
- Load the **Iris dataset** using `sklearn.datasets.load_iris()`.
- Store the data in a **Pandas DataFrame** with labeled columns.
- Split the dataset into **features (X) and target (y)**.
- Perform **train-test split (75% train, 25% test)**.
- Apply **feature scaling** using `StandardScaler()`.

### **2. Data Visualization**
- **Pairplot** to visualize feature relationships using `seaborn.pairplot()`.
- **Confusion Matrix heatmap** to analyze classification performance.
- **Feature importance bar plot** to identify influential features.

### **3. Model Training and Evaluation**
- Train a **Random Forest Classifier** with:
  - `n_estimators=120`
  - `max_depth=8`
  - `random_state=7`
- Predict species on the test set.
- Compute and display:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report**

---

## **Technologies Used**
- **Python**: Main programming language.
- **Libraries**:
  - `pandas`, `numpy`: Data handling.
  - `matplotlib`, `seaborn`: Data visualization.
  - `sklearn`: Machine learning model and evaluation tools.

---

## **Scope**
- Provides a **simple yet effective model** for flower classification.
- Can be extended to **different datasets** with minimal modifications.
- Helps understand **feature importance** and **model performance**.

---

## **Advantages**
- Easy implementation with **high accuracy**.
- Helps visualize **feature relationships** and **classification performance**.
- Random Forest provides **robust predictions** with feature importance analysis.

## **Disadvantages**
- Limited to the **Iris dataset**, which is **small and well-structured**.
- Random Forest may be **computationally expensive** for larger datasets.
- Feature importance scores might vary across different runs.

---

## **Future Improvements**
- Experiment with **different classifiers** (e.g., SVM, KNN, Neural Networks).
- Tune hyperparameters using **GridSearchCV** for better performance.
- Deploy the model using a **web interface** for real-time predictions.

---
## **Output**
<p align="center">
  <img src="https://github.com/user-attachments/assets/d1bb278a-82ed-4fd4-8146-4e7522751ed5" alt="Image 1">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d13e4352-61c7-4ce6-a34e-95194a429d25" alt="Image 2">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6b24a7f6-ef9c-45a3-9cab-9a3eab66e0d6" alt="Image 3">
</p>


### Contact
- **Name**: [Shriraj Dharmadhikari]
- **Company**: CODTECH IT SOLUTIONS
- **Email**: [shrirajdharmadhikari@gmail.com]
