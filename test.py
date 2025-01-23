import pandas as pd

# Step 1: Convert and save the file
df = pd.read_csv('data_reference.csv')

# Convert specific columns to float
df.iloc[:, 0] = df.iloc[:, 0].astype("float64")
df.iloc[:, 2] = df.iloc[:, 2].astype("float64")
df.iloc[:, 3] = df.iloc[:, 3].astype("float64")

# Save the intermediate file
df.to_csv('intermediate_data.csv', index=False)

# Step 2: Reload the file and perform calculations
df = pd.read_csv('intermediate_data.csv')

# Perform arithmetic operations
df.iloc[:, 0] *= 1000           # Multiply the first column by 1000
df.iloc[:, 2] /= 3.6           # Divide the third column by 3.6
df.iloc[:, 3] /= 3.6           # Divide the fourth column by 3.6

# Round the third and fourth columns to three decimal places
df.iloc[:, 2] = df.iloc[:, 2].round(3)
df.iloc[:, 3] = df.iloc[:, 3].round(3)

# Save the final processed file
df.to_csv('expert_data.csv', index=False)
