"""import matplotlib.pyplot as plt
import numpy as np

# Generate epochs
epochs = np.arange(1, 2001)

# Simulated loss values (keeping them similar to original trends)
train_loss = np.random.normal(loc=1.79, scale=0.005, size=len(epochs))
val_loss = np.random.normal(loc=1.79, scale=0.005, size=len(epochs))
train_loss[0] = 1.85  # Adding an initial peak for loss
val_loss[0] = 1.82

# Simulated accuracy (starting from 20% and gradually increasing to 98%)
train_accuracy = np.linspace(20, 98, len(epochs))
val_accuracy = np.linspace(20, 98, len(epochs))

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Loss
axes[0].plot(epochs, train_loss, label="Train Loss", color="blue")
axes[0].plot(epochs, val_loss, label="Validation Loss", color="orange")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Model Loss")
axes[0].legend()
axes[0].grid(True)

# Plot Accuracy
axes[1].plot(epochs, train_accuracy, label="Train Accuracy", color="blue")
axes[1].plot(epochs, val_accuracy, label="Validation Accuracy", color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Accuracy")
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(20, 100)  # Ensuring y-axis starts from 20 and does not exceed 100

# Display the plot
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Generate epochs
epochs = np.arange(1, 2001)

# Simulate accuracy starting from 20% and increasing to 98%
train_accuracy = np.linspace(20, 98, len(epochs))  
val_accuracy = np.linspace(20, 98, len(epochs))  

# Plot accuracy graph
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_accuracy, label="Train Accuracy", color="blue")
plt.plot(epochs, val_accuracy, label="Validation Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)
plt.ylim(20, 100)  # Ensure y-axis starts from 20 and does not exceed 100

plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Simulated data based on the original trend but modified to reach 98%
epochs = np.arange(1, 2001)
train_accuracy = np.linspace(20, 98, len(epochs))  # Gradually increasing accuracy
val_accuracy = np.linspace(20, 98, len(epochs))  # Validation accuracy follows a similar trend

# Plot the modified accuracy graph
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_accuracy, label="Train Accuracy", color="blue")
plt.plot(epochs, val_accuracy, label="Validation Accuracy", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)
plt.ylim(20, 100)  # Ensure y-axis starts from 20 and does not exceed 100

# Show the graph
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Generate epochs
epochs = np.arange(1, 2001)

# Simulated loss values (keeping them similar to original trends)
train_loss = np.random.normal(loc=1.79, scale=0.005, size=len(epochs))
val_loss = np.random.normal(loc=1.79, scale=0.005, size=len(epochs))
train_loss[0] = 1.85  # Adding an initial peak for loss
val_loss[0] = 1.82

# Simulated accuracy (starting from 0% and gradually increasing to 98%)
train_accuracy = np.linspace(0, 98, len(epochs))
val_accuracy = np.linspace(0, 98, len(epochs))

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Loss
axes[0].plot(epochs, train_loss, label="Train Loss", color="blue")
axes[0].plot(epochs, val_loss, label="Validation Loss", color="orange")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Model Loss")
axes[0].legend()
axes[0].grid(True)

# Plot Accuracy
axes[1].plot(epochs, train_accuracy, label="Train Accuracy", color="blue")
axes[1].plot(epochs, val_accuracy, label="Validation Accuracy", color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Accuracy")
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(0, 100)  # Ensuring y-axis starts from 0 and does not exceed 100

# Display the plot
plt.tight_layout()
plt.show()"""
import matplotlib.pyplot as plt
import numpy as np

# Generate epochs
epochs = np.arange(1, 2001)

# Simulated loss values (keeping them similar to original trends)
train_loss = np.random.normal(loc=1.79, scale=0.005, size=len(epochs))
val_loss = np.random.normal(loc=1.79, scale=0.005, size=len(epochs))
train_loss[0] = 1.85  # Adding an initial peak for loss
val_loss[0] = 1.82

# Simulated accuracy (following the original pattern but reaching 98%)
train_accuracy = np.zeros(len(epochs))
val_accuracy = np.zeros(len(epochs))

# Mimic the sharp rise in accuracy from the original image
train_accuracy[:10] = np.linspace(0, 40, 10)  # Initial jump to 40%
train_accuracy[10:50] = np.linspace(40, 48, 40)  # Gradual rise
train_accuracy[50:] = 98  # Stays at 98%

val_accuracy[:10] = np.linspace(0, 40, 10)
val_accuracy[10:50] = np.linspace(40, 48, 40)
val_accuracy[50:] = 98

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Loss
axes[0].plot(epochs, train_loss, label="Train Loss", color="blue")
axes[0].plot(epochs, val_loss, label="Validation Loss", color="orange")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Model Loss")
axes[0].legend()
axes[0].grid(True)

# Plot Accuracy
axes[1].plot(epochs, train_accuracy, label="Train Accuracy", color="blue")
axes[1].plot(epochs, val_accuracy, label="Validation Accuracy", color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Accuracy")
axes[1].legend()
axes[1].grid(True)
axes[1].set_ylim(0, 100)  # Ensuring y-axis starts from 0 and does not exceed 100

# Display the plot
plt.tight_layout()
plt.show()
