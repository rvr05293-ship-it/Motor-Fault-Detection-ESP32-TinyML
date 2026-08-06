# Real-Time Motor Fault Detection Using Unsupervised Autoencoders on ESP32 Edge Devices

## Overview

This project implements a TinyML-based motor fault detection system using an unsupervised autoencoder deployed on an ESP32 microcontroller. The model is trained using healthy motor data and detects abnormal motor conditions by measuring reconstruction error. The trained model is converted to TensorFlow Lite and deployed on the ESP32 for real-time edge inference.

---

## Features

- TinyML-based motor fault detection
- Autoencoder trained on healthy motor data
- TensorFlow Lite model deployment on ESP32
- Real-time Normal/Fault prediction
- Reconstruction error (MSE) based anomaly detection
- Lightweight edge AI implementation

---

## Tech Stack

- Python
- TensorFlow
- TensorFlow Lite
- C++
- Arduino IDE
- ESP32

---

## Repository Structure

```
Arduino/
    ESP32_Motor_AI.ino
    model.h

Python/
    train.py
    retrain.py
    train_esp32_compatible.py
    convert_quantized.py
    detect.py
    detect_novelty.py
    detect_org.py
    send_to_esp32.py

Images/
    system_workflow.png
    loss_distribution.png
    confusion_matrix.png
    esp32_hardware.jpg
```

---

## Workflow

1. Prepare and preprocess the motor dataset.
2. Train the autoencoder using healthy motor samples.
3. Convert the trained model into TensorFlow Lite format.
4. Deploy the quantized model on the ESP32.
5. Compute reconstruction error (MSE).
6. Compare the MSE against a predefined threshold.
7. Classify the input as **Normal** or **Fault**.

---

## System Architecture

![Workflow](Images/system_workflow.png)

---

## Loss Distribution

![Loss Distribution](Images/loss_distribution.png)

Healthy and faulty samples exhibit distinct reconstruction error distributions. A statistical threshold is used to separate normal and abnormal motor conditions.

---

## Confusion Matrix

![Confusion Matrix](Images/confusion_matrix.png)

The confusion matrix demonstrates the effectiveness of the selected threshold in distinguishing healthy and faulty motor samples.

---

## ESP32 Deployment

![ESP32 Hardware](Images/esp32_hardware.jpg)

The TensorFlow Lite model is deployed on the ESP32 microcontroller, where incoming motor data is classified as **Normal** or **Fault** using on-device inference.

---

## Future Improvements

- Live sensor integration
- OTA model updates
- Mobile dashboard
- Multi-fault classification

---

## Author

**Raghul V R**

B.Tech Electronics and Communication Engineering

Amrita Vishwa Vidyapeetham
