# IoT-Based Predictive Maintenance Using TinyML

An ESP32-based IoT predictive-maintenance system that uses TinyML and multiple sensors to detect equipment conditions in real time and automatically respond to critical faults.

## 📌 Project Overview

Traditional maintenance approaches often depend on scheduled inspections or waiting until equipment fails. This can result in unexpected downtime, equipment damage, and increased maintenance costs.

This project demonstrates a smart predictive-maintenance approach using **Embedded Systems, IoT, Machine Learning, and Automation**.

An ESP32 collects real-time sensor data from temperature, humidity, vibration, electrical current, and sound sensors. A TinyML model processes these measurements directly on the ESP32 and classifies the equipment condition into:

- 🟢 **NORMAL**
- 🟡 **WARNING**
- 🔴 **FAULT**

When a critical fault is detected, the system can automatically shut down the equipment using a relay and activate an alarm.

---

## 🎯 Objectives

- Monitor equipment condition in real time.
- Collect data from multiple physical sensors.
- Detect abnormal operating conditions.
- Train a machine-learning model for fault classification.
- Deploy the model on an ESP32 using TensorFlow Lite Micro.
- Perform inference directly on the edge device.
- Automatically shut down equipment during critical faults.
- Display system status locally.
- Provide a foundation for IoT-based predictive maintenance.

 System Architecture

                   ┌─────────────────────┐
                   │   Physical Equipment│
                   │       / Fan         │
                   └──────────┬──────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          ┌──────▼──────┐          ┌──────▼──────┐
          │   Sensors   │          │   Sensors   │
          │             │          │             │
          │ Temperature │          │  Vibration  │
          │ Humidity    │          │  Current    │
          │ Sound       │          │             │
          └──────┬──────┘          └──────┬──────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                       ┌──────▼──────┐
                       │    ESP32    │
                       │             │
                       │ Data        │
                       │ Processing  │
                       │ TinyML      │
                       └──────┬──────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐       ┌────▼─────┐
              │ Prediction │       │ Outputs  │
              │            │       │          │
              │ NORMAL     │       │ LCD      │
              │ WARNING    │       │ LEDs     │
              │ FAULT      │       │ Buzzer   │
              └────────────┘       │ Relay    │
                                   └───────────┘
