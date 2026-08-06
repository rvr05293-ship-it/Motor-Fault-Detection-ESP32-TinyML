import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


# =========================
# PARAMETERS
# =========================

SEGMENT_SIZE = 1024
EPOCHS = 60
BATCH_SIZE = 32


# =========================
# LOAD FILE FUNCTION
# =========================

def load_signal(filename):

    data = loadmat(filename)

    key = None

    for k in data.keys():

        if "DE_time" in k:

            key = k
            break

    signal = data[key].flatten()

    return signal


# =========================
# SEGMENT FUNCTION
# =========================

def segment_signal(signal):

    segments = []

    for i in range(0, len(signal) - SEGMENT_SIZE, SEGMENT_SIZE):

        segments.append(signal[i:i+SEGMENT_SIZE])

    return np.array(segments)


# =========================
# LOAD HEALTHY DATA
# =========================

print("Loading healthy data...")

files = [

"Normal_0.mat",
"Normal_1.mat",
"Normal_2.mat",
"Normal_3.mat"

]

healthy_segments = []

for file in files:

    signal = load_signal(file)

    seg = segment_signal(signal)

    healthy_segments.extend(seg)

healthy_segments = np.array(healthy_segments)

print("Total segments:", healthy_segments.shape)


# normalize

healthy_segments = healthy_segments / np.max(np.abs(healthy_segments))


# split

X_train, X_test = train_test_split(healthy_segments, test_size=0.2)


# =========================
# BUILD AUTOENCODER
# =========================

input_dim = SEGMENT_SIZE

input_layer = Input(shape=(input_dim,))

encoder = Dense(512, activation="relu")(input_layer)

encoder = Dense(128, activation="relu")(encoder)

encoder = Dense(32, activation="relu")(encoder)


decoder = Dense(128, activation="relu")(encoder)

decoder = Dense(512, activation="relu")(decoder)

decoder = Dense(input_dim, activation="linear")(decoder)


autoencoder = Model(input_layer, decoder)

autoencoder.compile(

optimizer=Adam(),

loss="mse"

)

autoencoder.summary()


# =========================
# TRAIN
# =========================

print("Training...")

history = autoencoder.fit(

X_train,
X_train,

epochs=EPOCHS,

batch_size=BATCH_SIZE,

validation_data=(X_test, X_test)

)


# =========================
# SAVE MODEL
# =========================

autoencoder.save("autoencoder_model.h5")

print("Training complete")