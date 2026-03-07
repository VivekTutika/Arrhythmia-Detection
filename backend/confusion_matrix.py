import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Define class labels
labels = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Ventricular Arrhythmia",
    "Conduction Block",
    "Premature Contraction",
    "ST Segment Abnormality"
]

# Define the modified confusion matrix
conf_matrix = np.array([
    [300, 5, 3, 4, 3, 2],
    [10, 110, 3, 4, 2, 1],
    [5, 3, 50, 2, 2, 2],
    [6, 3, 1, 50, 3, 2],
    [2, 1, 1, 2, 20, 2],
    [4, 1, 2, 3, 2, 23]
])

# Create a DataFrame for better plotting
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Modified Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
