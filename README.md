# 🧮 AI Handwritten Math Solver

> **Draw math. Get answers. Instantly.**

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-99.63%25-brightgreen?style=for-the-badge)

---
![WhatsApp Video 2026-03-11 at 22](https://github.com/user-attachments/assets/a0a8ead7-c7ef-446a-83c2-05250f84d10c)

## ✨ What It Does

Write any math expression by hand — just like on paper — and this AI reads it, understands it, and gives you the answer in real time.

No keyboard. No typing. Just draw.

---

## 🎬 How It Works

```
You draw on screen
        ↓
OpenCV detects each symbol
        ↓
CNN model reads what you wrote (99.63% accuracy)
        ↓
SymPy solves the expression
        ↓
Answer appears instantly
```

**Supports:**
- ➕ Addition &nbsp;&nbsp; `2 + 3 = 5`
- ➖ Subtraction &nbsp;&nbsp; `9 - 4 = 5`
- ✖️ Multiplication &nbsp;&nbsp; `6 * 7 = 42`
- ➗ Division &nbsp;&nbsp; `10 / 2 = 5`
- 🔢 Multi-digit numbers &nbsp;&nbsp; `12 + 34 = 46`
- 📐 Equations &nbsp;&nbsp; `2x + 3 = 7 → x = 2`

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/handwritten-math-solver
cd handwritten-math-solver
```

### 2. Set up environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Run the solver
```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **Draw** | Use mouse to write on the canvas |
| **S** | Solve the expression |
| **C** | Clear the canvas |
| **Q** | Quit |

---

## 🧠 How the AI Was Built

The model is a **Convolutional Neural Network (CNN)** trained from scratch on:
- **60,000 handwritten digit images** from the MNIST dataset
- **24,000 synthetic operator images** (custom generated for +, -, ×, ÷)

The architecture uses multiple Conv2D layers with BatchNormalization and Dropout for regularization — achieving **99.63% test accuracy**.

---

## 📁 Project Structure

```
handwritten-math-solver/
├── model/
│   ├── math_model.keras     ← Trained CNN model
│   └── classes.npy          ← Symbol class labels
├── main.py                  ← Live camera + detection app
├── train_model.py           ← CNN training script
├── solver.py                ← Math expression evaluator
└── requirements.txt
```

---

## 📦 Requirements

```
tensorflow
opencv-python
numpy
sympy
cvzone
matplotlib
scikit-learn
```

---

## 🏗️ Tech Stack

| Tool | Purpose |
|------|---------|
| **TensorFlow / Keras** | CNN model training & inference |
| **OpenCV** | Real-time drawing canvas & contour detection |
| **SymPy** | Mathematical expression solving |
| **NumPy** | Data processing & array operations |

---

## 👨‍💻 Author

**Saeed Fahim**
BTech — Artificial Intelligence & Machine Learning
Chandigarh University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/v-saeed-fahim)

---

> *Built as part of an AI/ML portfolio to demonstrate real-world deep learning applications.*
