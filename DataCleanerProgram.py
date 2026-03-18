import pandas as pd
import numpy as np


#Loading the raw data

df = pd.read_csav("cardio_train.csv", sep = ",")

print(f"Raw dataset shape: {df.shape}")
print(f"Raw dataset columns: {df.columns}\n")

