---
title: Titanic Survival Simulator
emoji: 🚢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🚢 Titanic: 3D Interactive Survival Simulator & Historical Experience

An interactive scientific and educational web application combining **Machine Learning models (Scikit-Learn)** with an **interactive 3D naval model (Plotly Mesh3d)**, deck position tracking, timeline simulation, and in-depth historical analysis of the factors that determined passenger survival during the sinking of the **R.M.S. Titanic (April 14–15, 1912)**.

---

## 🌟 Core Features

1. **🎟️ Step 1 — Boarding Pass Issuance & Passenger Identity**:
   - Choose from 10 real historical figures documented in the Kaggle dataset (e.g., Madeleine Force Astor, Margaret 'Molly' Brown, Lawrence Beesley, Jack Thayer, Master Trevor Allison, Sir Cosmo Duff Gordon, Anna Sofia Turja, Master Michel Navratil, Harold Goodwin, Apostolos Chronopoulos) or create a fully customized passenger profile.
   - Configure age, sex, social title (`Mr`, `Mrs`, `Miss`, `Master`), ticket class (1st, 2nd, 3rd), nationality, and accompanying companions (Solo, Pair, Small Family, Large Family).
   - Real-time conversion of 1912 British pound (£) fares into modern purchasing power equivalents in Euros (€).
   - Generation of an authentic vintage *White Star Line* boarding pass.

2. **🚢 Step 2 — 3D Ship Deck Locator & Interactive Model**:
   - Select deck location (Decks A through G) and ship side: **Port (Left — Officer Lightoller)** vs. **Starboard (Right — Officer Murdoch)**.
   - Interactive 3D wireframe and surface mesh model of the Titanic (hull, superstructure, 4 funnels, lifeboat deck).
   - Dynamic illuminated beacon marking the exact spatial position of the selected cabin.
   - Calculation of vertical distance (number of stair steps) and escape route distance to the boat deck.

3. **⏱️ Step 3 — The Night of the Sinking (Interactive Timeline Simulation)**:
   - Minute-by-minute timeline from iceberg collision (23:40) to final plunge (02:20) and rescue by RMS *Carpathia* (04:00).
   - Dynamic deck flooding monitoring tracking whether your cabin remains dry, becomes flooded, or has its escape routes cut off.
   - Real-time launch log of lifeboats by side and capacity utilization.

4. **🔮 Step 4 — Your Fate: Survival Probability & Explainable AI (XAI)**:
   - Prediction powered by an ensemble of Logistic Regression & Gradient Boosting Classifier models trained on the official dataset.
   - Dynamic survival probability gauge chart (0–100%) and binary outcome classification (Survived vs. Lost).
   - Feature Importance Waterfall breakdown explaining the exact mathematical contribution of passenger class, sex, age, family size, and ship side to the prediction.
   - Officer Protocol Differential: Murdoch's starboard protocol (men permitted when no women were waiting, ~37% male survival) vs. Lightoller's port protocol (strict "women and children only", ~18% male survival).

5. **📊 Step 5 — Key Notes, Research Findings & Forensics**:
   - **The Lifeboat Paradox**: Analysis of the 472 seats that went empty across the 20 lifeboats.
   - **Ticket Group Dynamics**: Statistical revelation that pairs had the highest survival rates (~53%), whereas large family groups collapsed below 16%.
   - **Socioeconomic Class Gap**: Disparity in survival between 1st, 2nd, and 3rd class passengers.
   - **Women & Children Protocol**: Near 100% survival rate for children in 1st/2nd class vs. ~42% in 3rd class.
   - **Forensic Science of the Disaster**: Watertight bulkheads overflow ("ice cube tray effect"), metallurgy of wrought iron rivet slag embrittlement in sub-zero water (-2.8°C / 27°F), and wireless telegraph communications.

---

## 📚 Academic & Historical References (Citations)

All statistical analyses, machine learning parameters, historical passenger records, and naval architecture specifications are grounded in officially recognized historical archives, government inquiry reports, and peer-reviewed literature (APA format):

### 1. Datasets & Historical Archives
- **Yasser, H. (2021).** *Titanic Dataset*. Kaggle.  
  Available at: [https://www.kaggle.com/datasets/yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Kaggle Inc. (2012).** *Titanic: Machine Learning from Disaster*. Kaggle Competition Archive.  
  Available at: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
- **Hind, P., et al. (1996–2026).** *Encyclopedia Titanica: Titanic Research, Passenger Manifests, Deck Plans & Biographies*.  
  Available at: [https://www.encyclopedia-titanica.org](https://www.encyclopedia-titanica.org)

### 2. Official Government Inquiries (1912)
- **British Wreck Commissioner's Court (Mersey Inquiry):**  
  Mersey, J. C. B., Baron. (1912). *Report on the Loss of the "Titanic" (S.S.): Formal Investigation into the Circumstances Attending the Foundering on 15th April, 1912, of the British Steamship "Titanic" of Liverpool*. Parliamentary Papers, Cd. 6352. London: His Majesty's Stationery Office.  
  Digital Archive: [British Inquiry Report - Titanic Inquiry Project](https://www.titanicinquiry.org/BOTInq/BOTReport/botRep01.php)
- **United States Senate Inquiry:**  
  Smith, W. A., et al. (1912). *"Titanic" Disaster: Hearings before a Subcommittee of the Committee on Commerce, United States Senate, Sixty-Second Congress, Second Session* (Senate Document No. 726 / Report No. 806). Washington, D.C.: Government Printing Office.  
  Digital Archive: [US Senate Titanic Report](https://www.titanicinquiry.org/USInq/USReport/AmInqRep01.php)

### 3. Survivor Eyewitness Memoirs
- **Beesley, L. (1912).** *The Loss of the SS Titanic: Its Story and Its Lessons*. Boston & New York: Houghton Mifflin Company.
- **Thayer, J. B., Jr. (1940).** *The Sinking of the S.S. Titanic: An Event That Shook the World*. Philadelphia: The Academy of Natural Sciences.
- **Lightoller, C. H. (1935).** *Titanic and Other Ships*. London: Ivor Nicholson and Watson.

### 4. Forensic & Naval Engineering Research
- **Foecke, T., & McCarty, J. (2008).** *What Really Sank the Titanic: New Forensic Discoveries*. New York: Citadel Press / Kensington Publishing Corp. (Metallurgical analysis of slag content embrittlement in cold seawater).
- **Garzke, W. H., Brown, D. K., & Woodward, J. B. (1997).** *The Structural Failure of the Titanic: Investigation and Analysis*. *SNAME Transactions* (Society of Naval Architects and Marine Engineers), 105, 237–264.

### 5. Scientific Software & Machine Learning Libraries
- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825–2830.
- **Plotly Technologies Inc. (2026).** *Plotly Open Source Graphing Library for Python*. [https://plotly.com/python/](https://plotly.com/python/)
- **Snowflake Inc. (2026).** *Streamlit: An open-source Python framework for data apps*. [https://streamlit.io/](https://streamlit.io/)

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application (5 Wizard Steps & 3D Visualizer)
├── Dockerfile                  # Container configuration for cloud deployment (port 7860)
├── requirements.txt            # Python dependencies (streamlit, scikit-learn, plotly, joblib, etc.)
├── titanic_model_bundle.joblib # Pre-trained Machine Learning models (Logistic Regression & GBDT)
├── Titanic.csv                 # Official passenger dataset
├── train_model.py              # Model retraining and bundle generation pipeline
└── README.md                   # Documentation, academic citations, and deployment metadata
```
