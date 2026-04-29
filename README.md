# 🛡️ Crowd Predictor Framework

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/Computer_Vision-YOLOv8-006400?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow_LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Prophet](https://img.shields.io/badge/Forecasting-Facebook_Prophet-257BD1?style=for-the-badge&logo=facebook&logoColor=white)](https://facebook.github.io/prophet/)

A professional-grade, real-time intelligence platform that combines **Computer Vision** and **Predictive Analytics** to monitor crowd density, ensure safety compliance, and forecast future risks.

---

## 🌟 Key Features

### 👁️ Intelligent Crowd Detection
- **Multi-Zone Monitoring:** Simultaneous tracking across 4 distinct zones (Zone A, B, C, D) with independent configurations.
- **Optimized YOLOv8:** Custom-tuned object detection pipeline for precise head and person tracking even in dense environments.
- **High Performance:** Real-time frame processing with asynchronous API communication.

### 📈 Predictive Intelligence
- **Hybrid Forecasting:** Combines **Multi-Variate LSTM** (Temporal patterns) and **Facebook Prophet** (Seasonality) for high-accuracy predictions (~85%+).
- **Risk Outlook:** Forecasts peak crowd sizes and risk durations to assist in proactive resource deployment.
- **Live Analytics:** Dynamic actual-vs-predicted trajectory graphs powered by Plotly.

### 🚨 Smart Alerting System
- **Real-time Thresholds:** Adjustable capacity limits per zone via the dashboard.
- **Automated Alerts:** Visual indicators and automated email notifications (via SMTP) when density exceeds safe thresholds.
- **Glassmorphism UI:** A premium, dark-themed dashboard built with Streamlit for a modern operator experience.

---

## 🛠️ Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | Streamlit, Plotly, Custom CSS (Glassmorphism) |
| **Computer Vision** | OpenCV, Ultralytics YOLOv8 |
| **Machine Learning** | TensorFlow (LSTM), Facebook Prophet, Pandas, NumPy |
| **Alerting** | SMTP (Email), Environment-based config |

---

## 📁 Project Structure

```bash
crowd_monitoring_system/
├── backend/                # FastAPI Application
│   ├── app/                # Core API logic (Routes, Services, Models)
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Streamlit Dashboard
│   ├── app.py              # Main UI Entry point
│   └── api.py              # API client logic
├── src/
│   └── ml/                 # Shared ML Model implementations (LSTM, Prophet)
├── data/                   # Datasets (Raw, Processed, Samples)
├── models/                 # Model weights (YOLO, Trained LSTMs)
├── notebooks/              # Research & Training experiments
├── tests/                  # Comprehensive test suite (Pytest)
├── .env                    # Environment variables (Credentials & Thresholds)
└── README.md               # You are here
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- Virtual Environment (recommended)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/m-shankar-m/crowd_monitoring_system.git
cd crowd_monitoring_system

# Install dependencies (Unified or per service)
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
# Email Alert Credentials
ALERT_EMAIL_TO=recipient@example.com
ALERT_EMAIL_FROM=sender@example.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=465

# Global Settings
MAX_CAPACITY=50
ALERT_EMAIL_COOLDOWN_SECONDS=60
```

---

## 💻 Running the Application

The system operates as a decoupled architecture. **Both services must be active.**

### Step 1: Start the Backend (API & AI Inference)
```bash
# From the root directory
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Start the Frontend (Dashboard)
```bash
# From the root directory in a new terminal
streamlit run frontend/app.py
```

---

## 🧪 Testing & Validation
The project includes a robust testing suite for ensuring reliability of detection and prediction logic.
```bash
# Run all tests
pytest tests/
```

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
