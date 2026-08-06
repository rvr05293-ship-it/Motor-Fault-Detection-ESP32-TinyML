import os
import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("model.h5", compile=False)

print("Model loaded successfully")


# --------------------------
# LOAD DATA FUNCTION
# --------------------------

def load_data(folder, label):

    data = []

    for file in os.listdir(folder):

        if file.endswith(".mat"):

            print("Loading:", file)

            mat = scipy.io.loadmat(os.path.join(folder, file))

            key = list(mat.keys())[-1]

            signal = mat[key].flatten()

            window_size = 100

            for i in range(0, len(signal) - window_size, window_size):

                window = signal[i:i+window_size]

                data.append(window)

    data = np.array(data)

    data = data.reshape((data.shape[0], data.shape[1], 1))

    labels = np.full(len(data), label)

    return data, labels


# --------------------------
# LOAD HEALTHY AND FAULT
# --------------------------

healthy_data, healthy_labels = load_data("healthy", 0)

fault_data, fault_labels = load_data("fault", 1)


# --------------------------
# PREDICT
# --------------------------

print("\nPredicting...")

healthy_pred = model.predict(healthy_data)

fault_pred = model.predict(fault_data)


# --------------------------
# RECONSTRUCTION ERROR
# --------------------------

healthy_error = np.mean(np.square(healthy_data - healthy_pred), axis=(1,2))

fault_error = np.mean(np.square(fault_data - fault_pred), axis=(1,2))


# --------------------------
# THRESHOLDS
# --------------------------

print("\nCalculating thresholds...")


threshold_meanstd = np.mean(healthy_error) + 3*np.std(healthy_error)

threshold_percentile = np.percentile(healthy_error, 95)

median = np.median(healthy_error)

mad = np.median(np.abs(healthy_error - median))

threshold_mad = median + 3*mad


# --------------------------
# EVALUATION FUNCTION
# --------------------------

def evaluate(name, threshold):

    healthy_pred_label = healthy_error > threshold
    fault_pred_label = fault_error > threshold

    y_true = np.concatenate([healthy_labels, fault_labels])
    y_pred = np.concatenate([healthy_pred_label, fault_pred_label])

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("\n===========================")
    print("METHOD:", name)

    print("Threshold:", threshold)

    print("Accuracy:", acc*100, "%")

    print("Confusion Matrix:")

    print(cm)

    print("===========================")


# --------------------------
# RUN ALL METHODS
# --------------------------

evaluate("Mean+3Std", threshold_meanstd)

evaluate("Percentile (95)", threshold_percentile)

evaluate("Median+MAD", threshold_mad)


print("\nPROJECT COMPLETED")