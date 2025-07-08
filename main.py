from preprocessing import load_and_prepare_data
from model import train_with_grid_search
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Load + scale data
    X_train_scaled, y_train, X_test_scaled, y_test, scaler, feature_names = load_and_prepare_data()

    # Train model (can be your GridSearch one or manual)
    model = train_with_grid_search(X_train_scaled, y_train, X_test_scaled, y_test)

    user_input = input("Enter 8 comma-separated values for: \n"
                       "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age:\n")
    
    # Parse and reshape
    input_arr = np.array([float(i) for i in user_input.split(',')]).reshape(1, -1)
    
    input_df = pd.DataFrame(input_arr, columns=feature_names)
    # Scale input like training data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)

    print(f"\nPrediction: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
    print(f"Probability of being diabetic: {proba[0][1] * 100:.2f}%")

