from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd

#real_data = load_demo()

real_data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\major_project\synthetic_heart_2020_3.csv")

real_data = real_data[0:10]
# Names of the columns that are discrete
discrete_columns = [
    'HeartDisease',
    'BMI',
    'Smoking',
    'AlcoholDrinking',
    'Stroke',
    'PhysicalHealth',
    'MentalHealth',
    'DiffWalking',
    'Sex',
    'AgeCategory',
    'Race',
    'Diabetic',
    'PhysicalActivity',
    'GenHealth',
    'SleepTime',
    'Asthma',
    'KidneyDisease',
    'SkinCancer'
]

ctgan = CTGAN(epochs=100)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(100)

df = pd.DataFrame(synthetic_data)

print(df)
