import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Confusion matrix values
# Format: [[TP, FP],
#          [FN, TN]]
cm = np.array([[110, 15],
               [20, 155]])

labels = ["Positive", "Negative"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", 
            xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"])

plt.title("Model Prediction Confusion Matrix\nAccuracy: 86.6%", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
