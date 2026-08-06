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

if not os.path.exists("results_novelty"):
    os.makedirs("results_novelty")


# ===============================
# LOAD MODEL
# ===============================

print("Loading model...")

model = load_model("autoencoder_model.h5", compile=False)

print("Model loaded successfully")


# ===============================
# LOAD SIGNAL FUNCTION
# ===============================

SEGMENT_SIZE = 1024

def load_signal(filename):

    mat = sio.loadmat(filename)

    signal = None

    for key in mat.keys():

        if "DE_time" in key:

            signal = mat[key].flatten()
            break


    segments = []

    for i in range(0, len(signal)-SEGMENT_SIZE, SEGMENT_SIZE):

        segments.append(signal[i:i+SEGMENT_SIZE])


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
# CALCULATE LOSS
# ===============================

healthy_loss = np.mean(np.square(healthy - healthy_pred), axis=1)

fault_loss = np.mean(np.square(fault - fault_pred), axis=1)



# ===============================
# THRESHOLDS (UPDATED HERE)
# ===============================

print("\nCalculating thresholds...")


thresholds = {

"Mean+3Std":
np.mean(healthy_loss) + 3*np.std(healthy_loss),


"Percentile(95)":
np.percentile(healthy_loss, 95),


"Median+MAD":
np.median(healthy_loss) + 3 * np.median(np.abs(healthy_loss - np.median(healthy_loss)))

}



# ===============================
# EVALUATE EACH METHOD
# ===============================

summary = ""

best_acc = 0
best_method = ""


for method, threshold in thresholds.items():

    print("\n==============================")

    print("METHOD:", method)

    print("Threshold:", threshold)


    healthy_pred_label = [0 if x<threshold else 1 for x in healthy_loss]

    fault_pred_label = [0 if x<threshold else 1 for x in fault_loss]


    healthy_true = [0]*len(healthy_pred_label)

    fault_true = [1]*len(fault_pred_label)


    y_true = healthy_true + fault_true

    y_pred = healthy_pred_label + fault_pred_label


    acc = accuracy_score(y_true,y_pred)

    cm = confusion_matrix(y_true,y_pred)


    TN, FP, FN, TP = cm.ravel()


    print("Accuracy:", acc*100,"%")

    print("False Positives:", FP)

    print("False Negatives:", FN)

    print("Confusion Matrix:\n", cm)


    if acc > best_acc:

        best_acc = acc
        best_method = method


    summary += "\nMETHOD: "+method+"\n"

    summary += "Threshold:"+str(threshold)+"\n"

    summary += "Accuracy:"+str(acc*100)+"\n"

    summary += "False Positives:"+str(FP)+"\n"

    summary += "False Negatives:"+str(FN)+"\n"

    summary += str(cm)+"\n"



    # SAVE CONFUSION MATRIX IMAGE

    plt.figure(figsize=(6,6))

    plt.imshow(cm)

    plt.title(method)

    plt.colorbar()

    plt.xlabel("Predicted")

    plt.ylabel("True")


    for i in range(2):

        for j in range(2):

            plt.text(j,i,cm[i,j],
            ha="center",
            va="center",
            color="white")


    plt.savefig("results_novelty/"+method+"_cm.png")

    plt.close()



# ===============================
# PRINT BEST METHOD
# ===============================

print("\n==============================")

print("BEST METHOD:", best_method)

print("BEST ACCURACY:", best_acc*100,"%")

print("==============================")


summary += "\nBEST METHOD:"+best_method+"\n"

summary += "BEST ACCURACY:"+str(best_acc*100)+"\n"



# ===============================
# SAVE SUMMARY FILE
# ===============================

with open("results_novelty/summary.txt","w") as f:

    f.write(summary)



# ===============================
# SAVE LOSS CSV
# ===============================

df = pd.DataFrame({

"Healthy_Loss": healthy_loss,

})

df.to_csv("results_novelty/healthy_loss.csv", index=False)



df2 = pd.DataFrame({

"Fault_Loss": fault_loss,

})

df2.to_csv("results_novelty/fault_loss.csv", index=False)



print("\nPROJECT COMPLETED")

print("Results saved in results_novelty folder")