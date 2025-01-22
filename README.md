# -EYE-STATE-PREDICTION-USING-EEG
 To find whether eye state can be predicted using EEG signals  collected from the brain through various sensors

 ### **Project Insight: EEG-Based Eye State Prediction**

This project focuses on predicting eye states (open or closed) using EEG (Electroencephalography) data. The goal is to classify the eye state accurately based on EEG signals collected from sensors placed on the brain. The project follows the standard machine learning workflow, ensuring a clean, understandable, and repeatable process.

---

### **Project Workflow**

1. **Problem Statement:**
   - Detect eye states (open or closed) from EEG signals, which can be useful in applications like gaming, assistive technologies, and medical research.

2. **Dataset Description:**
   - The dataset contains EEG readings from sensors with a target variable (`eyeDetection`) indicating whether eyes are open (`0`) or closed (`1`).
   - Key features include EEG readings from specific sensors such as `O1`, `P7`, `F7`, `F8`, `AF3`, and `AF4`.

3. **Preprocessing Steps:**
   - **Outlier Handling:** Removed or capped extreme values using methods like IQR-based capping and Winsorization.
   - **Skewness Correction:** Applied transformations like log and cube root to reduce skewness in features.
   - **Feature Scaling:** Standardized all features to ensure they are on the same scale.
   - **Feature Selection:** Selected the most relevant features using techniques like Random Forest importance.

4. **Data Balancing:**
   - Addressed class imbalance in the target variable using SMOTE (Synthetic Minority Oversampling Technique) to ensure both open and closed eye states were equally represented during training.

5. **Model Building:**
   - Built multiple machine learning models, including:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - Gaussian Naive Bayes
     - K-Nearest Neighbors (KNN)
   - Evaluated each model based on accuracy, precision, recall, F1-score, and confusion matrix.

6. **Hyperparameter Tuning:**
   - Optimized each model using GridSearchCV to find the best parameters, improving performance significantly.

7. **Best Model Selection:**
   - The **K-Nearest Neighbors (KNN)** classifier was selected as the best model with an accuracy of **88%** on test data.

8. **Testing with Unseen Data:**
   - Validated the model on unseen data, achieving consistent predictions for eye states.

---

### **Key Learnings**

- **Data Preprocessing:** Handling outliers and skewness is crucial to improve model performance.
- **Model Selection:** Hyperparameter tuning can significantly enhance model accuracy.
- **Feature Importance:** Using fewer, more relevant features reduces complexity without sacrificing accuracy.

---

### **Future Scope**

1. **Deploy the Model:**
   - Integrate the model into real-world applications like gaming or medical diagnostics.
2. **Explore Advanced Models:**
   - Experiment with deep learning models for potentially better accuracy.
3. **Continuous Improvement:**
   - Retrain the model periodically with new data to ensure relevance.

---

### **Conclusion**

This project successfully demonstrates the application of machine learning for EEG-based eye state prediction. It highlights the importance of data preprocessing, feature engineering, and model optimization in building an accurate and reliable classifier. The project provides a strong foundation for further exploration and application in real-world scenarios.
