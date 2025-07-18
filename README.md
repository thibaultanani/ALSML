# ALSML: Predicting ALS Progression and Survival with Machine Learning

[![DOI](https://zenodo.org/badge/DOI/10.1080/21678421.2025.2522399.svg)](https://doi.org/10.1080/21678421.2025.2522399)
[![Heroku App](https://img.shields.io/badge/WebApp-ALSML-blue?logo=heroku)](https://alsml-78a86daadd83.herokuapp.com/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](./LICENSE)

**ALSML** is a machine learning project for predicting disease progression and 1-year survival in patients with *Amyotrophic Lateral Sclerosis (ALS)* using clinical and functional features. The models are developed from real-world clinical data and support clinicians in patient stratification, prognosis, and personalized care planning.

🔗 **Live Demo**: [https://alsml-78a86daadd83.herokuapp.com/](https://alsml-78a86daadd83.herokuapp.com/)  
⚠️ *Please note that the hosting server may change soon. This link may be updated in the near future.*

---

## 🧠 Overview

This project implements and compares various machine learning models to:

- Predict 1-year survival in ALS patients.
- Forecast ALS Functional Rating Scale (ALSFRS) scores over 12 months.
- Identify key clinical features using metaheuristic-based feature selection.

We leverage:
- **Logistic Regression**, **Random Forests**, **K Nearest Neighbors**, etc.
- **Differential Evolution (DE)** to optimize feature subset selection.
- **Kaplan–Meier survival analysis** and **clustering** for patient stratification.

This work is detailed in our publication:

📄 *T. Anani et al., "Feature selection using metaheuristics to predict annual amyotrophic lateral sclerosis progression", ALS & Frontotemporal Degeneration, 2025.*  
🔗 [PubMed](https://pubmed.ncbi.nlm.nih.gov/40621723/) | [DOI](https://doi.org/10.1080/21678421.2025.2522399)

---

## 📊 Datasets

We use three real-world ALS datasets:

- **PRO-ACT**: The largest public ALS clinical trials dataset (4,659 patients).
- **ExonHit**: Clinical trial data (384 patients).
- **PULSE**: A multicenter French cohort (198 patients, used for independent validation).

These datasets include demographic, clinical, and functional features such as ALSFRS, BMI, onset site, pulmonary function (FVC), blood pressure, and derived staging scores (King’s, MiToS, FT9).

---

## 🛠️ Features

- 🧬 Predict 1-year survival probability.
- 📈 Predict ALSFRS scores at T3, T6, T9, and T12 using baseline data.
- 📉 Select informative features using Differential Evolution (DE).
- 📊 Interpret model decisions with SHAP values and survival curves.
- 🖥️ Use models directly from a **user-friendly web interface**.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/thibaultanani/ALSML.git
cd ALSML
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
pip install -r requirements.txt
```

---

## 🧪 Results Summary

- **Best model (survival)**: Logistic Regression + Differential Evolution
  - Balanced Accuracy: **76.3%**
  - AUC: **0.84**
  - C-index: **0.8**

- **Best model (ALSFRS evolution)**: LightGBM
  - RMSE (T3 to T12): **3.16 to 5.37**
  - R²: **0.78 to 0.55**
  - PCC: **0.88 to 0.74**

- **Key predictive features**: FVC, age, BMI, ALSFRS decline rate, and King's stage.

---

## 🖥️ Web App

Try the models with your own data via our online application:

🔗 **Web Application**: [alsml-78a86daadd83.herokuapp.com](https://alsml-78a86daadd83.herokuapp.com/)

Features:
- Upload patient features by filling in the form.
- Interpret results with charts and risk levels.
- Download a summary file to save results.

Note: The application is hosted on Heroku and may migrate to another server soon.

---

## 📚 Citation

If you use this work in your research, please cite our article:

```bibtex
@article{anani2025alsml,
  title = {Feature selection using metaheuristics to predict annual amyotrophic lateral sclerosis progression},
  author = {Anani, Thibault and Pradat-Peyre, Jean-François and Delbot, François and Desnuelle, Claude and Rolland, Anne-Sophie and Devos, David and Pradat, Pierre-François},
  journal = {Amyotrophic Lateral Sclerosis and Frontotemporal Degeneration},
  year = {2025},
  doi = {10.1080/21678421.2025.2522399}
}
```

---

## 📬 Contact

For questions or contributions:

- **Thibault Anani** — thibault.anani-agondja@lip6.fr  
- Laboratory: **LIP6, Sorbonne Université**

---

## 📄 License

This project is licensed under the BSD 3-Clause License — see the [LICENSE](./LICENSE) file for details.
