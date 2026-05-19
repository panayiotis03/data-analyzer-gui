# 📊 data-analyzer-gui

Μια διαδραστική εφαρμογή γραφικού περιβάλλοντος (GUI) σε web περιβάλλον, αναπτυγμένη με **Streamlit**.

Η εφαρμογή αναλύει την συσχέτιση των δεδομένών περιβαλλοντικών αισθητήρων (IAQ - Indoor Air Quality) και αντικειμενικών δεδομένών των φοιτητών (feedback) από 3 του διαφορετικές αίθουσες του ΤΕΧΝΟΛΟΓΙΚΟ ΠΑΝΕΠΙΣΤΗΜΙΟ ΚΥΠΡΟΥ, και παράγει πλήρες ανάλυση, αυτοματοποιημένες εισηγήσεις βελτίωσης καθώς και αναφορές.

---

# Οδηγίες Εγκατάστασης & Εκτέλεσης

# 1️⃣ Προαπαιτούμενα
Πριν ξεκινήσεις, βεβαιώσου ότι έχεις εγκατεστημένα:
- Python 3.8 ή νεότερη έκδοση
- pip (Python package manager)

# 2️⃣ Εγκατάσταση βιβλιοθηκών
Άνοιξε το Terminal ή Command Prompt και εκτέλεσε:
pip install streamlit pandas matplotlib numpy reportlab

# 3️⃣ Λήψη του Project
Κατέβασε ή κάνε clone το repository και μπες στον φάκελο όπου βρίσκετε το suggestions_analyzer
cd path/to/suggestions_analyzer

# 4️⃣ Εκκίνηση της εφαρμογής
Τρέξε την εφαρμογή με την εντολή:
streamlit run suggestions_analyzer.py

# 5️⃣ Άνοιγμα στο browser
Αφού γίνει η εκκίνηση, η εφαρμογή θα ανοίξει αυτόματα στο:
http://localhost:8501

# Δομή Αρχείων Δεδομένων
Η αναμενόμενη δομή φακέλων για τη λειτουργία του συστήματος:

data/
├── Tofis/
│   ├── sep.csv, oct.csv, nov.csv, dec.csv, jan.csv      <- Sensor data
│   ├── TOFIS_Sep_feedbacks.csv                          <- Feedback data
│   └── TOFIS_Oct_feedbacks.csv  
├── drakos/
│   ├── drakos front/  ->  sep.csv, oct.csv  ...
│   ├── drakos back/   ->  sep.csv, oct.csv  ...
│   ├── DRAKOS_Front_Oct_feedbacks.csv
│   └── DRAKOS_Back_Nov_feedbacks.csv  ...
└── tasos/
    ├── TASOS_front_left/  ->  sep.csv, oct.csv  ...
    ├── TASOS_back_right/    sep.csv, oct.csv  ...
    ├── TASOS_Front_Left_Oct_feedbacks.csv
    └── TASOS_Back_Right_Jan_feedbacks.csv  ...
