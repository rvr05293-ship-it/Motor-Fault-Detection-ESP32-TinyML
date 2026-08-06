import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score


# ===============================
# CREATE RESULTS FOLDER
# ===============================

if not os.path.exists("results"):
    os.makedirs("results")


# ===============================
# PARAMETERS
# ===============================

SEGMENT_SIZE = 1024


# ===============================
# LOAD MODEL
# ===============================

print("Loading model...")

model = load_model("autoencoder_model.h5", compile=False)

print("Model loaded successfully")


# ===============================
# LOAD SIGNAL FUNCTION
# ===============================

def load_signal(filename):

    mat = sio.loadmat(filename)

    signal = None

    for key in mat.keys():

        if "DE_time" in key:

            signal = mat[key].flatten()

            break


    segments = []

    for i in range(0, len(signal)-SEGMENT_SIZE, SEGMENT_SIZE):

        segment = signal[i:i+SEGMENT_SIZE]

        segments.append(segment)


    return np.array(segments)



# ===============================
# LOAD HEALTHY DATA
# ===============================

healthy_segments = []

for file in os.listdir():

    if "Normal" in file and file.endswith(".mat"):

        print("Loading healthy:", file)

        healthy_segments.append(load_signal(file))


healthy = np.vstack(healthy_segments)



# ===============================
# LOAD FAULT DATA
# ===============================

fault_segments = []

for file in os.listdir():

    if file.startswith("B") and file.endswith(".mat"):

        print("Loading fault:", file)

        fault_segments.append(load_signal(file))


fault = np.vstack(fault_segments)



# ===============================
# PREDICT
# ===============================

print("\nPredicting...")

healthy_pred = model.predict(healthy)

fault_pred = model.predict(fault)



# ===============================
# RECONSTRUCTION LOSS
# ===============================

healthy_loss = np.mean(np.square(healthy - healthy_pred), axis=1)

fault_loss = np.mean(np.square(fault - fault_pred), axis=1)



# ===============================
# ADAPTIVE THRESHOLD (NOVELTY)
# ===============================

threshold = np.mean(healthy_loss) + 3*np.std(healthy_loss)

print("\nAdaptive Threshold:", threshold)



# ===============================
# DETECT FAULTS
# ===============================

print("\nFault Detection Results:\n")

fault_result = []

for i, loss in enumerate(fault_loss):

    if loss > threshold:

        print("Sample", i+1, "→ FAULT")

        fault_result.append("FAULT")

    else:

        print("Sample", i+1, "→ HEALTHY")

        fault_result.append("HEALTHY")



# ===============================
# ACCURACY
# ===============================

healthy_pred_label = [0 if x<threshold else 1 for x in healthy_loss]

fault_pred_label = [0 if x<threshold else 1 for x in fault_loss]


healthy_true = [0]*len(healthy_pred_label)

fault_true = [1]*len(fault_pred_label)


y_true = healthy_true + fault_true

y_pred = healthy_pred_label + fault_pred_label


acc = accuracy_score(y_true,y_pred)

print("\nAccuracy:", acc*100, "%")



# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_true,y_pred)

print("\nConfusion Matrix:\n", cm)



# ===============================
# SAVE CONFUSION MATRIX IMAGE
# ===============================

plt.figure(figsize=(6,6))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

for i in range(2):

    for j in range(2):

        plt.text(j,i,cm[i,j],
                 ha="center",
                 va="center",
                 color="white",
                 fontsize=16)

plt.savefig("results/confusion_matrix.png")

plt.close()



# ===============================
# SAVE RESULTS CSV
# ===============================

df = pd.DataFrame({

"Loss": fault_loss,

"Prediction": fault_result

})

df.to_csv("results/fault_results.csv", index=False)



# ===============================
# SAVE SUMMARY FILE
# ===============================

with open("results/summary.txt","w") as f:

    f.write("Adaptive Threshold: " + str(threshold) + "\n")

    f.write("Accuracy: " + str(acc*100) + "\n")

    f.write("Confusion Matrix:\n")

    f.write(str(cm))



print("\n===================================")
print("PROJECT SUCCESSFULLY COMPLETED")
print("Results saved in results folder")
print("===================================")