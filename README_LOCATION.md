# 📍 Project Location Guide

## Industrial IoT Multivariate Anomaly Detection Pipeline

---

### ✅ Correct Project Structure

```
/home/lucas-junges/Documents/material_estudo/projetos/
│
├── projeto 1/                                    # Projeto 1
│
├── projeto 2/                                    # ⭐ THIS PROJECT (IoT Anomaly Detection)
│   ├── README.md                                 # Full documentation
│   ├── PROJECT_SUMMARY.md                        # Quick reference
│   ├── COMPLETION_SUMMARY.md                     # Completion details
│   ├── requirements.txt                          # Dependencies
│   ├── train_simple.py                          # Training script
│   ├── evaluate_simple.py                       # Evaluation script
│   │
│   ├── src/                                     # Source code
│   │   ├── data_generation/                     # IoT simulator
│   │   ├── preprocessing/                       # Feature engineering
│   │   ├── models/                              # ML models
│   │   ├── evaluation/                          # Metrics & KPIs
│   │   └── utils/                               # Visualization
│   │
│   ├── data/                                    # Generated data (28+ MB)
│   │   ├── raw/                                 # Sensor data (30 days)
│   │   ├── processed/                           # Train/test splits
│   │   └── results/                             # Evaluation reports
│   │
│   ├── models/                                  # Trained models
│   │   ├── preprocessor.pkl
│   │   ├── isolation_forest/
│   │   ├── lof/
│   │   └── autoencoder/
│   │
│   └── docs/                                    # Documentation
│       ├── technical_report.pdf                 # 21-page LaTeX report
│       ├── technical_report.tex
│       └── deployment_guide.md
│
└── lucaswilliamjunges-website/                  # Portfolio Website
    ├── index.html                               # Main portfolio (project card added)
    ├── projects/
    │   └── iot-anomaly-detection.html          # Project showcase page
    └── ...
```

---

### 🎯 Quick Access

#### 1. **Full Project (Projeto 2)**
```bash
cd "/home/lucas-junges/Documents/material_estudo/projetos/projeto 2"
```

**Contains:**
- ✅ Complete source code
- ✅ Trained models (Isolation Forest, LOF, Autoencoder)
- ✅ Generated data (43,200 samples, 30 days)
- ✅ Technical documentation (PDF + Markdown)
- ✅ Training & evaluation scripts

#### 2. **Portfolio Website Reference**
```bash
cd "/home/lucas-junges/Documents/material_estudo/projetos/lucaswilliamjunges-website"
```

**Contains:**
- ✅ `index.html` - Main portfolio with project card
- ✅ `projects/iot-anomaly-detection.html` - Project showcase page

---

### 🚀 How to Use

#### Run the complete pipeline:
```bash
cd "/home/lucas-junges/Documents/material_estudo/projetos/projeto 2"

# Install dependencies
pip install -r requirements.txt

# Train all models (IF, LOF, Autoencoder)
python train_simple.py

# Run evaluation
python evaluate_simple.py

# View results
cat data/results/evaluation_report.txt

# View technical report
xdg-open docs/technical_report.pdf
```

#### View in portfolio:
```bash
# Open portfolio website
cd "/home/lucas-junges/Documents/material_estudo/projetos/lucaswilliamjunges-website"
firefox index.html  # or your browser

# Navigate to Projects section → Industrial IoT Anomaly Detection
```

---

### 📊 What's Where

| Content | Location | Description |
|---------|----------|-------------|
| **Main Project** | `projeto 2/` | All code, data, models, docs |
| **Source Code** | `projeto 2/src/` | Python modules |
| **Trained Models** | `projeto 2/models/` | Pickled models + weights |
| **Generated Data** | `projeto 2/data/` | Raw + processed datasets |
| **Documentation** | `projeto 2/docs/` | LaTeX report + guides |
| **Portfolio Card** | `lucaswilliamjunges-website/index.html` | Project showcase |
| **Project Page** | `lucaswilliamjunges-website/projects/iot-anomaly-detection.html` | Detailed view |

---

### ✅ Verification Checklist

Run these commands to verify everything is in place:

```bash
# Check project files
ls "/home/lucas-junges/Documents/material_estudo/projetos/projeto 2"

# Check data files
ls -lh "/home/lucas-junges/Documents/material_estudo/projetos/projeto 2/data/raw/"

# Check trained models
ls -lh "/home/lucas-junges/Documents/material_estudo/projetos/projeto 2/models/"

# Check documentation
ls "/home/lucas-junges/Documents/material_estudo/projetos/projeto 2/docs/"

# Check portfolio integration
grep -n "Industrial IoT" "/home/lucas-junges/Documents/material_estudo/projetos/lucaswilliamjunges-website/index.html"
```

---

### 📝 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Comprehensive technical documentation |
| `PROJECT_SUMMARY.md` | Quick reference guide |
| `COMPLETION_SUMMARY.md` | Project completion details |
| `train_simple.py` | Train all 3 models from scratch |
| `evaluate_simple.py` | Run comprehensive evaluation |
| `docs/technical_report.pdf` | 21-page academic-quality report |
| `docs/deployment_guide.md` | Edge vs cloud architecture |

---

### 🎓 For Portfolio Presentation

**When showcasing this project:**

1. **Start with:** Portfolio website (`lucaswilliamjunges-website/index.html`)
2. **Click:** "Industrial IoT Anomaly Detection" project card
3. **Show:** Project metrics (94.3% detection, 59x ROI, 8.2 min TTD)
4. **Deep dive:** Open `projeto 2/docs/technical_report.pdf`
5. **Demo (optional):** Run `python train_simple.py` to show pipeline

---

**All set! ✅** Your project is now properly organized in `projeto 2/` with portfolio integration in the website.
