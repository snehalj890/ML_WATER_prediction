import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load("water_quality_model.pkl")

print("Searching for parameter combinations classified as Potable (1) by your model...")

np.random.seed(42)
found = False

for i in range(5000):
    # Generate realistic random water parameter values
    ph = np.random.uniform(6.5, 8.5)
    hardness = np.random.uniform(150, 250)
    solids = np.random.uniform(15000, 35000)
    chloramines = np.random.uniform(4, 10)
    sulfate = np.random.uniform(250, 400)
    conductivity = np.random.uniform(300, 500)
    organic_carbon = np.random.uniform(8, 16)
    trihalomethanes = np.random.uniform(40, 90)
    turbidity = np.random.uniform(2, 5)

    sample_df = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate, 
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]], columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ])

    prediction = model.predict(sample_df)[0]
    
    if prediction == 1:
        print(f"\n✅ Found a working Potable sample combination after {i+1} tries!")
        print("Copy these exact numbers into your Streamlit app inputs:")
        for col, val in sample_df.iloc[0].items():
            print(f"  {col}: {val:.2f}")
        found = True
        break

if not found:
    print("\n❌ Tested 5,000 random combinations and the model predicted 0 for all. This confirms your model has an extreme class bias toward 'Not Potable' due to how it was trained.")