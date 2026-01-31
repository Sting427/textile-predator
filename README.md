# ROTex™ | Advanced Textile Operating System

![Version](https://img.shields.io/badge/version-v20.0-blue) ![Status](https://img.shields.io/badge/status-production--ready-success) ![Python](https://img.shields.io/badge/python-3.9%2B-yellow) ![License](https://img.shields.io/badge/license-Enterprise-black)

**ROTex** is a cloud-native Industrial Command Center designed for the modern textile manufacturing sector. It integrates **Real-time Market Intelligence**, **Computer Vision (QC)**, **IoT Telemetry**, and **Financial Forecasting** into a unified executive dashboard.

---

## 🚀 System Modules

### 1. 📡 War Room (Market Intelligence)
* **Live Price Tracking:** Real-time API feeds for Cotton (NYMEX), Gas (Henry Hub), and Yarn Fair Value.
* **Predictive AI:** Linear Regression models forecast raw material costs 7 days into the future.
* **Intel Stream:** Automated aggregation of global textile news and threat detection.
* **Reporting:** One-click generation of PDF Executive Summaries with embedded trend charts.

### 2. 🧪 Virtual Laboratory (QC)
* **GSM Master:** Automated fabric weight calculation and grading (Light/Medium/Heavy).
* **Shrinkage Simulator:** Calculates dimensional stability against international 5% tolerance standards.
* **AQL Inspector:** Generates sampling plans based on **ISO 2859-1 (AQL 2.5)** for shipment auditing.

### 3. 🏭 Factory IoT (Digital Twin)
* **Live Telemetry:** Simulates real-time sensor data from the production floor (Loom Speed, Humidity, Power).
* **Alert System:** Triggers visual alarms when environmental conditions (e.g., Temperature > 34°C) exceed safety thresholds.

### 4. 👁️ Vision AI (Automated Inspection)
* **Defect Detection:** Uses OpenCV to analyze fabric images and identify weaving faults.
* **Auto-Logging:** Automatically records Pass/Fail results into the SQL database for audit trails.

### 5. 💰 Deal Breaker (Financial Engine)
* **Margin Calculator:** Instantly computes net profit per kg based on live yarn costs and overheads.
* **Secure Ledger:** Commits all transaction details to a persistent SQLite database.

---

## 🛠️ Tech Stack

* **Core Engine:** Python 3.9
* **Frontend:** Streamlit (Responsive / Mobile-First Design)
* **Database:** SQLite3 (Transaction-Safe Storage)
* **Computer Vision:** OpenCV (cv2)
* **Data Science:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Plotly Interactive, PyDeck 3D Geospatial
* **Reporting:** FPDF (PDF Generation), Matplotlib

---

## 📱 Mobile Architecture
ROTex features a **Responsive UI** that adapts to any device.
* **Desktop:** Full-width analytical dashboards.
* **Mobile:** Collapsed sidebars, touch-optimized buttons, and stacked metric cards for on-the-floor usage.

---

## 🔐 Security Protocol
The system is protected by a **Level-1 Authentication Gateway**.
* **Access Control:** Password-protected entry.
* **Audit Logging:** All financial and QC actions are immutably recorded in `rotex_core.db`.

---

> *"The future of textile manufacturing is not just in the machinery, but in the data."*
> **— ROTex Systems**
