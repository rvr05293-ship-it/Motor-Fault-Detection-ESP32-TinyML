---

## Sample Detection Results

### Healthy Motor and Ball Fault

The model successfully distinguishes healthy motor data from ball bearing fault data based on reconstruction error during real-time inference.

![Healthy and Ball Fault](Images/result1.jpeg)

---

### Inner Race Fault and Outer Race Fault

The system also detects different fault categories, including inner race and outer race faults, demonstrating the effectiveness of the trained autoencoder in identifying abnormal motor conditions.

![Inner and Outer Race Fault](Images/result2.jpeg)

---

## ESP32 Deployment

The TensorFlow Lite model is deployed on the ESP32 microcontroller, where motor condition data is processed and classified as **Normal** or **Fault** using on-device inference.

---
