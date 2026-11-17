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

Μια διαδραστική επιστημονική και εκπαιδευτική εφαρμογή (Interactive Web Application & Historical Simulation) βασισμένη στην τραγωδία του υπερωκεανίου **R.M.S. Titanic (14-15 Απριλίου 1912)**. 

Η εφαρμογή συνδυάζει **μοντέλα Μηχανικής Μάθησης (Machine Learning)** με **τρισδιάστατη διαδραστική απεικόνιση (3D Plotly Mesh)**, αναπαράσταση της θέσης του επιβάτη στο πλοίο, καθώς και εξειδικευμένη ιστορική και στατιστική ανάλυση των παραγόντων που καθόρισαν την επιβίωση ή την απώλεια.

---

## 🌟 Κύρια Χαρακτηριστικά (Core Features)

1. **🎟️ Βήμα 1 - Έκδοση Εισιτηρίου & Ταυτότητα Επιβάτη**:
   - Δυνατότητα επιλογής ανάμεσα σε 10 πραγματικά ιστορικά πρόσωπα του Kaggle dataset (π.χ. Madeleine Force Astor, Margaret 'Molly' Brown, Lawrence Beesley, Jack Thayer, Master Trevor Allison, Sir Cosmo Duff Gordon, Anna Sofia Turja, Master Michel Navratil, Harold Goodwin, Απόστολος Χρονόπουλος) ή πλήρους διαμόρφωσης εξατομικευμένου επιβάτη.
   - Επιλογή ηλικίας, βιολογικού φύλου, κοινωνικού τίτλου (`Mr`, `Mrs`, `Miss`, `Master`), θέσης/τάξης (1η, 2η, 3η), εθνικότητας και συνοδών (Solo, Pair, Small Family, Large Family).
   - Αυτόματος υπολογισμός και μετατροπή ναύλου σε λίρες 1912 και σύγχρονη αγοραστική αξία ευρώ (€).
   - Παραγωγή αυθεντικού vintage boarding pass της *White Star Line*.

2. **🚢 Βήμα 2 - 3D Προσομοίωση Θέσης στο Σκάφος (Interactive 3D Deck Locator)**:
   - Επιλογή καταστρώματος (Decks A έως G) και πλευράς του πλοίου: **Port (Αριστερά - Lightoller)** vs **Starboard (Δεξιά - Murdoch)**.
   - Τρισδιάστατο διαδραστικό μοντέλο του Τιτανικού (κύτος, υπερκατασκευές, 4 καπνοδόχοι, κατάστρωμα λέμβων).
   - Φωτεινός παλμικός δείκτης (Beacon) που αποτυπώνει την ακριβή χωρική θέση της καμπίνας.
   - Υπολογισμός απόστασης και υψομετρικής διαφοράς (αριθμός σκαλοπατιών) μέχρι το κατάστρωμα των σωσίβιων λέμβων.

3. **⏱️ Βήμα 3 - Η Νύχτα του Ναυαγίου (Interactive Timeline)**:
   - Χρονοδιάγραμμα των κρίσιμων στιγμών από τη σύγκρουση (23:40) μέχρι τη βύθιση (02:20) και την άφιξη του *Carpathia* (04:00).
   - Παρακολούθηση της στάθμης του νερού ανά κατάστρωμα σε πραγματικό χρόνο και αξιολόγηση της προσβασιμότητας των εξόδων διαφυγής.
   - Ζωντανή απεικόνιση των σωσιβίων λέμβων που καθελκύονται ανά πλευρά του πλοίου.

4. **🔮 Βήμα 4 - Η Μοίρα σας: Πιθανότητα Επιβίωσης & ML Επεξήγηση (Explainable AI)**:
   - Πρόβλεψη επιβίωσης με συνδυασμό αλγορίθμων Logistic Regression & Gradient Boosting Classifier.
   - Μετρητής πιθανότητας (Gauge Chart 0-100%) και ταξινόμηση έκβασης (Επιβίωση / Απώλεια).
   - Επεξήγηση συνεισφοράς χαρακτηριστικών (Feature Importance Waterfall) αναλύοντας πώς η κοινωνική τάξη, το φύλο, η ηλικία και η πλευρά του πλοίου διαμόρφωσαν το τελικό αποτέλεσμα.
   - Επίδραση του πρωτοκόλλου των Αξιωματικών: First Officer Murdoch (Starboard - επιβίωση ανδρών ~37%) vs Second Officer Lightoller (Port - επιβίωση ανδρών ~18%).

5. **📊 Βήμα 5 - Key Notes, Ερευνητικά Στατιστικά & Επιστήμη Ναυαγίου**:
   - **Το Παράδοξο των Λέμβων**: Ανάλυση των 472 κενών θέσεων στις 20 λέμβους.
   - **Ticket Groups**: Στατιστική ανάλυση των κοινών εισιτηρίων (υψηλή επιβίωση ζευγαριών ~53% vs κατάρρευση πολύτεκνων οικογενειών <16%).
   - **Ταξικό Χάσμα**: Ανάλυση ανισότητας μεταξύ 1ης, 2ης και 3ης θέσης.
   - **Πρωτόκολλο 'Γυναίκες & Παιδιά'**: Σύγκριση επιβίωσης παιδιών 1ης/2ης θέσης (σχεδόν 100%) έναντι 3ης θέσης (μόλις ~42%).
   - **Η Επιστήμη του Ναυαγίου**: Bulkheads (διαμερίσματα), μεταλλουργική αστοχία ήλων (rivet slag embrittlement), το ζήτημα του ασυρμάτου του *SS Californian* και θερμοδυναμική υποθερμίας στους -2.8°C.

---

## 📚 Βιβλιογραφία & Επιστημονικές Αναφορές (References & Citations)

Όλες οι αναλύσεις, οι παράμετροι του μοντέλου μηχανικής μάθησης, τα ιστορικά δεδομένα των επιβατών και οι προδιαγραφές του πλοίου βασίζονται στις παρακάτω επίσημες πηγές και επιστημονικές δημοσιεύσεις (μορφότυπο APA):

### 1. Σύνολα Δεδομένων & Πλατφόρμες Δεδομένων (Datasets)
- **Yasser, H. (2021).** *Titanic Dataset*. Kaggle.  
  Διαθέσιμο στο: [https://www.kaggle.com/datasets/yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Kaggle Inc. (2012).** *Titanic: Machine Learning from Disaster*. Kaggle Competition Archive.  
  Διαθέσιμο στο: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
- **Hind, P., et al. (1996–2026).** *Encyclopedia Titanica: Titanic Research, Passenger Manifests, Deck Plans & Biographies*.  
  Διαθέσιμο στο: [https://www.encyclopedia-titanica.org](https://www.encyclopedia-titanica.org)

### 2. Επίσημες Κρατικές Ανακρίσεις (Official Government Inquiry Reports, 1912)
- **British Wreck Commissioner's Court (Mersey Inquiry):**  
  Mersey, J. C. B., Baron. (1912). *Report on the Loss of the "Titanic" (S.S.): Formal Investigation into the Circumstances Attending the Foundering on 15th April, 1912, of the British Steamship "Titanic" of Liverpool*. Parliamentary Papers, Cd. 6352. London: His Majesty's Stationery Office.  
  Ψηφιακό αρχείο: [British Inquiry Report - Titanic Inquiry Project](https://www.titanicinquiry.org/BOTInq/BOTReport/botRep01.php)
- **United States Senate Inquiry:**  
  Smith, W. A., et al. (1912). *"Titanic" Disaster: Hearings before a Subcommittee of the Committee on Commerce, United States Senate, Sixty-Second Congress, Second Session* (Senate Document No. 726 / Report No. 806). Washington, D.C.: Government Printing Office.  
  Ψηφιακό αρχείο: [US Senate Titanic Report](https://www.titanicinquiry.org/USInq/USReport/AmInqRep01.php)

### 3. Αυτόπτες Μάρτυρες & Απομνημονεύματα Επιζώντων (Eyewitness Memoirs)
- **Beesley, L. (1912).** *The Loss of the SS Titanic: Its Story and Its Lessons*. Boston & New York: Houghton Mifflin Company.
- **Thayer, J. B., Jr. (1940).** *The Sinking of the S.S. Titanic: An Event That Shook the World*. Philadelphia: The Academy of Natural Sciences.
- **Lightoller, C. H. (1935).** *Titanic and Other Ships*. London: Ivor Nicholson and Watson.

### 4. Εγκληματολογικές & Μεταλλουργικές Μελέτες (Forensic & Naval Architecture Studies)
- **Foecke, T., & McCarty, J. (2008).** *What Really Sank the Titanic: New Forensic Discoveries*. New York: Citadel Press / Kensington Publishing Corp. (Μεταλλουργική ανάλυση περιεκτικότητας σκωρίας σε ήλους/πριτσίνια σφυρήλατου σιδήρου).
- **Garzke, W. H., Brown, D. K., & Woodward, J. B. (1997).** *The Structural Failure of the Titanic: Investigation and Analysis*. *SNAME Transactions* (Society of Naval Architects and Marine Engineers), 105, 237–264.

### 5. Επιστημονικό Λογισμικό & Βιβλιοθήκες (Scientific Software & Libraries)
- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825–2830.
- **Plotly Technologies Inc. (2026).** *Plotly Open Source Graphing Library for Python*. [https://plotly.com/python/](https://plotly.com/python/)
- **Snowflake Inc. (2026).** *Streamlit: An open-source Python framework for data apps*. [https://streamlit.io/](https://streamlit.io/)

---

## 📁 Δομή Αρχείων Εφαρμογής (Repository Structure)

```text
├── app.py                     # Κύριος κώδικας της εφαρμογής Streamlit (5 Wizard Steps & 3D Visualizer)
├── Dockerfile                 # Αρχείο containerization για εκτέλεση σε cloud υποδομές (port 7860)
├── requirements.txt           # Εξαρτήσεις Python (streamlit, scikit-learn, plotly, joblib κ.ά.)
├── titanic_model_bundle.joblib# Προ-εκπαιδευμένα μοντέλα (Logistic Regression & Gradient Boosting)
├── Titanic.csv                # Το επίσημο σύνολο δεδομένων επιβατών
├── train_model.py             # Script επανεκπαίδευσης και εξαγωγής μοντέλου
└── README.md                  # Τεκμηρίωση, επιστημονικές αναφορές και metadata
```
