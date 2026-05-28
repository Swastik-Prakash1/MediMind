<div align="center">

<br/>

<!-- AWARD BANNER -->
<img src="https://img.shields.io/badge/🏆%20%201st%20Place-%20Build%20with%20Gemini%20×%20MLH%20%7C%20Zetech%20HackDay%20International%202026-FFD700?style=for-the-badge&labelColor=1a1a2e" alt="Award"/>

<br/><br/>

<h1>
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Stethoscope.png" alt="Stethoscope" width="50" align="center"/>
  &nbsp;MediMind
</h1>

<h3>AI-Powered Clinical Triage & Emergency Companion</h3>

<p><em>Context-aware. Multimodal. Zero-latency SOS. Built to save lives.</em></p>

<br/>

<!-- LIVE DEMO — MOST PROMINENT ELEMENT -->
<a href="https://medimind-0ciq.onrender.com" target="_blank">
  <img src="https://img.shields.io/badge/%F0%9F%94%B4%20%20LIVE%20DEMO%20%E2%80%94%20Click%20to%20Open%20App-FF0000?style=for-the-badge&logoColor=white&labelColor=8B0000" alt="Live Demo" height="50"/>
</a>

<br/><br/>

> ### 👉 [**medimind-0ciq.onrender.com**](https://medimind-0ciq.onrender.com)
> *Live. Interactive. No login required.*

<br/>

<!-- DEMO VIDEO PLACEHOLDER — replace src with your actual video thumbnail URL -->
<a href="https://medimind-0ciq.onrender.com" target="_blank">
  <img src="https://img.shields.io/badge/▶%20%20Watch%20Demo%20Video-000000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch Demo Video"/>
</a>

<br/><br/>

<!-- BADGES -->
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=flat-square&logo=flask&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini%20AI-4285F4?style=flat-square&logo=google&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-43.5%25-F7DF1E?style=flat-square&logo=javascript&logoColor=black)
![Status](https://img.shields.io/badge/Status-Live%20%26%20Deployed-00C851?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

</div>

---

## 🎯 What is MediMind?

**MediMind** is not another symptom chatbot. It's a **multimodal AI health intelligence system** that bridges the gap between a patient describing symptoms and a doctor receiving structured, actionable clinical data.

Where standard health apps treat every query in isolation, MediMind:

- **Remembers** — builds a temporal symptom timeline across sessions
- **Sees** — analyzes uploaded images of physical symptoms using vision AI
- **Guards** — bypasses the LLM entirely for zero-latency emergency detection
- **Reports** — generates professional S.O.A.P clinical handoff documents automatically

```
Patient says "I had a fever yesterday and now I have a rash."
→ Standard chatbot: Answers each symptom independently
→ MediMind: Connects the timeline → probable viral infection → routes to Infectious Disease specialist
```

---

## 💡 The Problem We Solved

| Problem | How MediMind Fixes It |
|---|---|
| Symptom checkers treat queries in isolation | Temporal memory engine connects symptom timelines |
| Patients can't describe invisible symptoms | Multimodal image upload for visual diagnosis |
| LLM response latency = death in emergencies | Guardian Mode bypasses AI with direct SOS trigger |
| Doctors get incomplete patient history | Auto-generated SOAP clinical handoff report |
| Healthcare AI leaks sensitive data | Privacy-first local JSON storage — no cloud dependency |

---

## 🌟 Key Features

### 🧠 1. Context-Aware Triage (Temporal Logic)

MediMind maintains a **persistent memory module** that tracks the full progression of a patient's symptoms over time.

- *Yesterday: fever. Today: stomach pain.* → The AI connects these as a timeline, not two isolated events.
- Implemented via **recursive prompt engineering with JSON history injection** — every LLM call includes the full structured conversation context.

---

### 📸 2. Multimodal Visual Diagnosis

Patients can **upload photos** of visible symptoms — rashes, wounds, swelling, discoloration.

- The image is analyzed alongside the full text transcript using **Google Gemini's multimodal vision model**
- Enables specialist routing decisions: *"This rash pattern suggests Dermatology, not General Practice"*

---

### 🚨 3. Guardian Mode — Zero-Latency SOS Protocol

A **passive semantic listener** that runs before the LLM call — not after it.

**Trigger phrases:** `"chest pain"` · `"numb arm"` · `"can't breathe"` · `"seizure"` · `"unconscious"`

**Immediate actions:**
1. UI locks into **Red Alert Mode** instantly
2. 10-second countdown timer activates
3. High-frequency audio distress beacon emits
4. Emergency services contact screen appears

> The entire SOS response is triggered in **< 50ms** — no LLM round-trip involved.

---

### 📋 4. Automated Clinical Handoff (SOAP Report)

One click generates a professional **S.O.A.P. note** — the standard format used by medical professionals worldwide:

| Section | Content |
|---|---|
| **S**ubjective | Patient-reported symptoms, timeline, severity |
| **O**bjective | Analyzed indicators, image findings, risk score |
| **A**ssessment | AI-ranked probable conditions with confidence |
| **P**lan | Specialist recommendation + urgency level |

---

### 🎨 5. Risk Stratification Engine

Every triage session is scored and color-coded:

| Level | Color | Meaning |
|---|---|---|
| Low | 🟢 Green | Self-manageable, monitor at home |
| Moderate | 🟡 Yellow | Schedule a doctor visit soon |
| High | 🔴 Red | Seek medical attention urgently |

Risk factors: symptom severity, duration, frequency, emotional tone, and patient history.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PATIENT INPUT                           │
│            Voice / Text / Image Upload                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FLASK BACKEND                              │
│                                                              │
│   ┌─────────────────┐    ┌──────────────────────────────┐   │
│   │  GUARDIAN MODE  │    │      CONTEXT ENGINE           │   │
│   │  (SOS Check)    │    │  Fetch history from JSON DB   │   │
│   │  < 50ms trigger │    │  Inject into LLM prompt       │   │
│   └────────┬────────┘    └──────────────┬───────────────┘   │
│            │ Critical?                  │                    │
│            ▼ YES                        ▼                    │
│   🚨 RED ALERT MODE          ┌──────────────────────┐       │
│   Audio Beacon + Timer       │   GEMINI AI (LLM +   │       │
│                              │   Vision Multimodal) │       │
│                              └──────────┬───────────┘       │
│                                         │                    │
│                              ┌──────────▼───────────┐       │
│                              │  SOAP Report Engine  │       │
│                              │  Risk Scorer         │       │
│                              └──────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
              Frontend Dashboard (HTML/CSS/JS)
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML5 · CSS3 · Vanilla JS | Responsive UI, real-time updates, SOS overlay |
| **Backend** | Python 3.9+ · Flask | API routing, context injection, report generation |
| **AI / LLM** | Google Gemini (multimodal) | Triage reasoning + image analysis |
| **Memory** | Local JSON storage | Privacy-first session persistence |
| **Deployment** | Render.com | Live production hosting |
| **Audio** | Web Audio API | SOS distress beacon generation |

---

## 📁 Project Structure

```
MediMind/
├── backend/
│   ├── app.py              # Flask application entry point
│   ├── triage.py           # Core triage + LLM orchestration
│   ├── guardian.py         # SOS keyword detection engine
│   ├── soap_report.py      # SOAP note generation
│   └── risk_engine.py      # Risk stratification scoring
├── frontend/
│   ├── index.html          # Main application UI
│   ├── style.css           # Styling + Guardian Mode red alert
│   └── app.js              # Client logic, voice, image upload
├── data.json               # Local patient history store
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9+
- A Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey))

### 1. Clone the Repository

```bash
git clone https://github.com/Swastik-Prakash1/MediMind.git
cd MediMind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Create a .env file in the root directory
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 4. Run the Application

```bash
cd backend
python app.py
```

Open your browser at `http://localhost:5000`

---

## 🧪 How to Test (Demo Scenarios)

| Scenario | What to Do | Expected Behavior |
|---|---|---|
| **Temporal Memory** | Report a fever. Wait. Report a rash. | AI connects both symptoms as a timeline |
| **Visual Diagnosis** | Upload a photo of a skin rash | AI recommends Dermatology specialist |
| **Guardian SOS** | Type "I have chest pain and numb arm" | Instant Red Alert + countdown + beacon |
| **SOAP Report** | Click "Generate Report" after triage | Professional clinical handoff document |
| **Risk Scoring** | Describe severe, multi-day symptoms | Red risk indicator in dashboard |

---

## 🏆 Recognition

<div align="center">

**🥇 1st Place — Build with Gemini × MLH**
**Zetech HackDay International 2026**

*Awarded for: Multimodal AI integration · Temporal clinical memory · Zero-latency emergency protocol*

</div>

---

## ⚠️ Medical Disclaimer

MediMind is an **AI-assisted healthcare support tool** — not a medical device, certified diagnostic system, or replacement for professional medical advice. Always consult a licensed healthcare provider for diagnosis and treatment decisions.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss your idea before submitting a PR.

```bash
# Fork → Clone → Create feature branch → PR
git checkout -b feature/your-feature-name
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🩺 by [Swastik Prakash](https://github.com/Swastik-Prakash1)

⭐ **Star this repo if MediMind impressed you!**

[![Live Demo](https://img.shields.io/badge/Try%20It%20Live-medimind--0ciq.onrender.com-FF0000?style=for-the-badge)](https://medimind-0ciq.onrender.com)

</div>
