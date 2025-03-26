import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def download_data():
    """Download the dataset if not available."""
    url = "https://raw.githubusercontent.com/your-repo/autism_data.csv"  # Replace with actual dataset URL
    if not os.path.exists("autism_data.csv"):
        r = requests.get(url)
        with open("autism_data.csv", "wb") as f:
            f.write(r.content)

def load_data():
    """Load and preprocess data."""
    download_data()
    data = pd.read_csv("autism_data.csv")
    data.dropna(inplace=True)
    data_classes = data['Class/ASD'].apply(lambda x: 1 if x == 'YES' else 0)
    features = data[['age', 'result']]
    scaler = MinMaxScaler()
    features[['age', 'result']] = scaler.fit_transform(features)
    return features, data_classes

def train_models(X_train, y_train):
    """Train different models and return them in a dictionary."""
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=5, random_state=1),
        'SVM': SVC(kernel='linear', C=1, gamma=2, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=10),
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def main():
    st.title("Autism Spectrum Disorder Prediction")
    st.write("Enter details to predict the likelihood of ASD.")
    
    # User Input
    age = st.slider("Age", 0, 100, 25)
    result = st.slider("Screening Test Score", 0, 10, 5)
    model_choice = st.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes", "Logistic Regression"])
    
    # Load and Train Models
    X, y = load_data()
    models = train_models(X, y)
    
    # Normalize Input
    scaler = MinMaxScaler()
    scaler.fit(X[['age', 'result']])
    input_data = scaler.transform(np.array([[age, result]]))
    
    # Predict
    if st.button("Predict"):
        model = models[model_choice]
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1] if hasattr(model, 'predict_proba') else [0]
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"The model predicts a high likelihood of ASD. Confidence: {probability[0]*100:.2f}%")
        else:
            st.success(f"The model predicts a low likelihood of ASD. Confidence: {(1 - probability[0])*100:.2f}%")

if __name__ == "__main__":
    main()
