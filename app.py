import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & LUXURY WHITE STAR LINE STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic: 3D Survival Simulator & Historical Experience",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_str: str):
    """
    Renders HTML safely in Streamlit without CommonMark / Markdown
    converting indented lines or empty lines into <pre><code> code blocks.
    """
    lines = [line.strip() for line in html_str.strip().split("\n") if line.strip()]
    clean_html = "\n".join(lines)
    st.markdown(clean_html, unsafe_allow_html=True)

# Custom CSS for rich maritime look
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .brand-title {
        font-family: 'Cinzel', serif;
        letter-spacing: 1px;
    }

    .main {
        background: radial-gradient(circle at 50% 10%, #0d1b2a 0%, #080f1a 100%);
        color: #e0e6ed;
    }

    .boarding-pass {
        background: linear-gradient(135deg, #162438 0%, #0d1726 100%);
        border: 2px solid #d4af37;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
        margin-top: 10px;
    }

    .boarding-header {
        font-family: 'Cinzel', serif;
        font-size: 1.35rem;
        color: #ffd166;
        margin-bottom: 2px;
    }

    .boarding-sub {
        font-size: 0.8rem;
        color: #a4b8d1;
        letter-spacing: 2px;
        border-bottom: 1px solid rgba(212,175,55,0.3);
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    .pass-label {
        font-size: 0.72rem;
        color: #829ab8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .pass-val {
        font-size: 1.05rem;
        color: #ffffff;
        font-weight: 700;
    }

    .info-card {
        background: rgba(18, 30, 49, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
    }

    .highlight-gold {
        color: #d4af37;
        font-weight: 700;
    }

    .highlight-cyan {
        color: #48cae4;
        font-weight: 600;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.25s ease;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Cinzel', serif;
        color: #ffd166;
    }
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL BUNDLE & DATASET
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    bundle_path = os.path.join(app_dir, "model", "titanic_model_bundle.joblib")
    if not os.path.exists(bundle_path):
        bundle_path = os.path.join(app_dir, "titanic_model_bundle.joblib")
    if not os.path.exists(bundle_path):
        import subprocess
        train_script = os.path.join(app_dir, "train_model.py")
        subprocess.run(["python3", train_script], check=True)
        bundle_path = os.path.join(app_dir, "model", "titanic_model_bundle.joblib")
        if not os.path.exists(bundle_path):
            bundle_path = os.path.join(app_dir, "titanic_model_bundle.joblib")
    
    bundle = joblib.load(bundle_path)
    return bundle

bundle = load_model_bundle()
lr_model = bundle["lr_model"]
gb_model = bundle["gb_model"]
feature_cols = bundle["feature_columns"]
coefficients = bundle["coefficients"]
intercept = bundle["intercept"]
df_clean = bundle["df_clean"]

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL LIFEBOAT DATABASE
# ─────────────────────────────────────────────────────────────────────────────
LIFEBOATS = [
    {"boat": "7", "side": "Starboard", "time": "00:45", "capacity": 65, "loaded": 28, "officer": "Murdoch", "type": "Standard"},
    {"boat": "5", "side": "Starboard", "time": "00:55", "capacity": 65, "loaded": 41, "officer": "Murdoch", "type": "Standard"},
    {"boat": "6", "side": "Port",      "time": "00:55", "capacity": 65, "loaded": 28, "officer": "Lightoller", "type": "Standard"},
    {"boat": "3", "side": "Starboard", "time": "01:00", "capacity": 65, "loaded": 50, "officer": "Murdoch", "type": "Standard"},
    {"boat": "1", "side": "Starboard", "time": "01:10", "capacity": 40, "loaded": 12, "officer": "Murdoch", "type": "Emergency Cutter"},
    {"boat": "8", "side": "Port",      "time": "01:10", "capacity": 65, "loaded": 39, "officer": "Lightoller", "type": "Standard"},
    {"boat": "10", "side": "Port",     "time": "01:20", "capacity": 65, "loaded": 55, "officer": "Lightoller", "type": "Standard"},
    {"boat": "9", "side": "Starboard", "time": "01:20", "capacity": 65, "loaded": 56, "officer": "Murdoch", "type": "Standard"},
    {"boat": "12", "side": "Port",     "time": "01:25", "capacity": 65, "loaded": 42, "officer": "Lightoller", "type": "Standard"},
    {"boat": "11", "side": "Starboard","time": "01:25", "capacity": 65, "loaded": 70, "officer": "Murdoch", "type": "Standard"},
    {"boat": "14", "side": "Port",     "time": "01:30", "capacity": 65, "loaded": 63, "officer": "Lowe / Lightoller", "type": "Standard"},
    {"boat": "16", "side": "Port",     "time": "01:35", "capacity": 65, "loaded": 56, "officer": "Lightoller", "type": "Standard"},
    {"boat": "13", "side": "Starboard","time": "01:35", "capacity": 65, "loaded": 64, "officer": "Murdoch", "type": "Standard"},
    {"boat": "15", "side": "Starboard","time": "01:35", "capacity": 65, "loaded": 70, "officer": "Murdoch", "type": "Standard"},
    {"boat": "C", "side": "Starboard", "time": "01:40", "capacity": 47, "loaded": 71, "officer": "Murdoch", "type": "Collapsible"},
    {"boat": "2", "side": "Port",      "time": "01:45", "capacity": 40, "loaded": 26, "officer": "Lightoller", "type": "Emergency Cutter"},
    {"boat": "4", "side": "Port",      "time": "01:55", "capacity": 65, "loaded": 40, "officer": "Lightoller", "type": "Standard"},
    {"boat": "D", "side": "Port",      "time": "02:05", "capacity": 47, "loaded": 44, "officer": "Lightoller", "type": "Collapsible"},
    {"boat": "B", "side": "Port",      "time": "02:15", "capacity": 47, "loaded": 25, "officer": "Overturned", "type": "Collapsible"},
    {"boat": "A", "side": "Starboard", "time": "02:15", "capacity": 47, "loaded": 12, "officer": "Swamped", "type": "Collapsible"},
]
df_boats = pd.DataFrame(LIFEBOATS)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL PERSONAS FROM KAGGLE DATASET (yasserh/titanic-dataset)
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "⚙️ Προσαρμοσμένος Επιβάτης (Custom Passenger)": {
        "name": "", "age": 28, "sex": "male", "pclass": 2, "title": "Mr",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Solo (Μόνος/η)", "deck": "D", "side": "Starboard (Δεξιά - Murdoch)",
        "fare": 13.0, "dataset_id": None, "ticket": "", "actual_survived": None, "boat": "",
        "insight": ""
    },
    "👑 Madeleine Force Astor (1η Θέση - 18 ετών, Σύζυγος J.J. Astor IV, Λέμβος 4) [ID #701]": {
        "name": "Madeleine Talmadge Force Astor", "age": 18, "sex": "female", "pclass": 1, "title": "Mrs",
        "nationality": "American (USA)", "embarked": "Cherbourg (C)", "companions": "Pair (2 άτομα - π.χ. με σύντροφο/φίλο)", "deck": "C", "side": "Port (Αριστερά - Lightoller)",
        "fare": 227.52, "dataset_id": 701, "ticket": "PC 17757", "actual_survived": 1, "boat": "Λέμβος 4 (Port)",
        "insight": "Η 18χρονη έγκυος σύζυγος του συνταγματάρχη John Jacob Astor IV (του πλουσιότερου ανθρώπου στον πλανήτη). Ο Astor τη βοήθησε να επιβιβαστεί στη Λέμβο 4 από το παράθυρο του καταστρώματος περιπάτου. Όταν ρώτησε τον υποπλοίαρχο Lightoller αν μπορούσε να επιβιβαστεί για να την προστατεύσει λόγω της προχωρημένης εγκυμοσύνης της, ο Lightoller αρνήθηκε κατηγορηματικά («Όχι άνδρες μέχρι να επιβιβαστούν όλες οι γυναίκες»). Ο Astor χάθηκε στα παγωμένα νερά, ενώ η Madeleine σώθηκε και γέννησε τον γιο τους John Jacob Astor VI λίγους μήνες αργότερα."
    },
    "💎 Margaret 'Molly' Brown (1η Θέση - 44 ετών, 'The Unsinkable Molly Brown', Λέμβος 6) [ID #195]": {
        "name": "Margaret 'Molly' Tobin Brown", "age": 44, "sex": "female", "pclass": 1, "title": "Mrs",
        "nationality": "American (USA)", "embarked": "Cherbourg (C)", "companions": "Solo (Μόνος/η)", "deck": "B", "side": "Port (Αριστερά - Lightoller)",
        "fare": 27.72, "dataset_id": 195, "ticket": "PC 17610", "actual_survived": 1, "boat": "Λέμβος 6 (Port)",
        "insight": "Η παγκοσμίως διάσημη «Αβύθιστη Μόλι Μπράουν». Στη Λέμβο 6 (που αναχώρησε με μόλις 24 άτομα αντί για 65), απαίτησε από τον πηδαλιούχο Robert Hichens να γυρίσουν πίσω για να σώσουν ναυαγούς. Όταν εκείνος αρνήθηκε έντρομος, η Brown πήρε μόνη της το κουπί και κράτησε τις γυναίκες ζεστές με τα γουναρικά της. Στο διασωστικό πλοίο Carpathia μοίρασε τρόφιμα και ίδρυσε την Επιτροπή Επιζώντων, συγκεντρώνοντας $10,000 για τους άπορους μετανάστες της 3ης θέσης."
    },
    "💼 Prof. Lawrence Beesley (2η Θέση - 34 ετών, Καθηγητής Φυσικής Cambridge, Λέμβος 13) [ID #22]": {
        "name": "Lawrence Beesley", "age": 34, "sex": "male", "pclass": 2, "title": "Mr",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Solo (Μόνος/η)", "deck": "D", "side": "Starboard (Δεξιά - Murdoch)",
        "fare": 13.0, "dataset_id": 22, "ticket": "248698", "actual_survived": 1, "boat": "Λέμβος 13 (Starboard)",
        "insight": "Καθηγητής Φυσικής στο Dulwich College και απόφοιτος Cambridge. Βρισκόταν στη δεξιά πλευρά (Starboard) υπό τον Αξιωματικό William Murdoch, ο οποίος εφήρμοζε το «Γυναίκες και παιδιά πρώτα» (όχι αποκλειστικά), επιτρέποντας σε άνδρες να μπουν όταν δεν υπήρχαν γυναίκες στην ουρά. Καθώς η Λέμβος 13 κατέβαινε, ο Murdoch του φώναξε να πηδήξει καθώς υπήρχε κενή θέση. Ο Beesley συνέγραψε το διασημότερο και ακριβέστερο ιστορικό χρονικό: 'The Loss of the SS Titanic' (1912)."
    },
    "🎓 Jack Thayer Jr. (1η Θέση - 17 ετών, Αναποδογυρισμένη Πτυσσόμενη B) [ID #551]": {
        "name": "John Borland 'Jack' Thayer Jr.", "age": 17, "sex": "male", "pclass": 1, "title": "Mr",
        "nationality": "American (USA)", "embarked": "Cherbourg (C)", "companions": "Small Group (3-4 άτομα - Οικογένεια)", "deck": "C", "side": "Port (Αριστερά - Lightoller)",
        "fare": 110.88, "dataset_id": 551, "ticket": "17421", "actual_survived": 1, "boat": "Πτυσσόμενη B (Αναποδογυρισμένη)",
        "insight": "17χρονος γιος του Αντιπροέδρου των Σιδηροδρόμων Πενσυλβάνια John B. Thayer. Αφού αποχωρίστηκε τους γονείς του στο χάος των τελευταίων λεπτών, πήδηξε στον παγωμένο Ατλαντικό και κατάφερε να σκαρφαλώσει στην αναποδογυρισμένη πτυσσόμενη λέμβο B μαζί με τον υποπλοίαρχο Lightoller. Ο πατέρας του πνίγηκε, αλλά η μητέρα του Marian (#582) σώθηκε. Οι μαρτυρίες και τα σκίτσα του Jack Thayer ήταν η πρώτη αδιάσειστη επιβεβαίωση ότι ο Τιτανικός κόπηκε στα δύο πριν βυθιστεί."
    },
    "👶 Master Hudson Trevor Allison (1η Θέση - Βρέφος 11 μηνών, Λέμβος 11) [ID #306]": {
        "name": "Master Hudson Trevor Allison", "age": 1, "sex": "male", "pclass": 1, "title": "Master",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Small Group (3-4 άτομα - Οικογένεια)", "deck": "C", "side": "Starboard (Δεξιά - Murdoch)",
        "fare": 151.55, "dataset_id": 306, "ticket": "113781", "actual_survived": 1, "boat": "Λέμβος 11 (Starboard)",
        "insight": "Βρέφος μόλις 11 μηνών (0.92 ετών στο dataset). Μέσα στον πανικό, η νταντά του Alice Cleaver τον πήρε και επιβιβάστηκε στη Λέμβο 11. Οι εύποροι γονείς του και η 2χρονη αδελφούλα του Helen (#298) αρνήθηκαν να επιβιβαστούν αναζητώντας απεγνωσμένα το βρέφος τους στο πλοίο, χωρίς να γνωρίζουν ότι ήταν ήδη ασφαλές. Ολόκληρη η οικογένεια χάθηκε. Η μικρή Helen Allison ήταν το μοναδικό παιδί 1ης και 2ης θέσης που έχασε τη ζωή του στον Τιτανικό."
    },
    "🤺 Sir Cosmo Edmund Duff Gordon (1η Θέση - 49 ετών, 'The Money Boat', Λέμβος 1) [ID #600]": {
        "name": "Sir Cosmo Edmund Duff Gordon", "age": 49, "sex": "male", "pclass": 1, "title": "Rare",
        "nationality": "British (UK)", "embarked": "Cherbourg (C)", "companions": "Pair (2 άτομα - π.χ. με σύντροφο/φίλο)", "deck": "A", "side": "Starboard (Δεξιά - Murdoch)",
        "fare": 56.93, "dataset_id": 600, "ticket": "PC 17485", "actual_survived": 1, "boat": "Έκτακτη Λέμβος Cutter 1 (Starboard)",
        "insight": "Σκωτσέζος βαρονέτος και Ολυμπιονίκης ξιφασκίας. Επιβιβάστηκε με τη σύζυγό του Lady Duff Gordon στη Λέμβο 1 (χωρητικότητας 40 ατόμων), η οποία αναχώρησε με μόλις 12 επιβαίνοντες (7 μέλη πληρώματος και 5 επιβάτες 1ης θέσης). Ο Duff Gordon έδωσε φιλοδώρημα £5 σε κάθε ναύτη, προκαλώντας τεράστιο σκάνδαλο στον Τύπο («The Money Boat»), κατηγορούμενος ότι δωροδόκησε το πλήρωμα για να μην επιστρέψει να σώσει ναυαγούς."
    },
    "☘️ Anna Sofia Turja (3η Θέση - 18 ετών, Φινλανδή Μετανάστρια, Λέμβος 15) [ID #678]": {
        "name": "Anna Sofia Turja", "age": 18, "sex": "female", "pclass": 3, "title": "Miss",
        "nationality": "Scandinavian (Σουηδία/Νορβηγία)", "embarked": "Southampton (S)", "companions": "Solo (Μόνος/η)", "deck": "F", "side": "Starboard (Δεξιά - Murdoch)",
        "fare": 9.84, "dataset_id": 678, "ticket": "4138", "actual_survived": 1, "boat": "Λέμβος 15 (Starboard)",
        "insight": "18χρονη αγρότισσα από τη Φινλανδία που ταξίδευε ολομόναχη για το Οχάιο, χωρίς να γνωρίζει λέξη αγγλικά. Παρά το γεγονός ότι οι περισσότερες γυναίκες της 3ης θέσης εγκλωβίστηκαν στα χαμηλά καταστρώματα λόγω κλειδωμένων θυρών και γλώσσας, η Anna ακολούθησε μια ομάδα συμπατριωτών της που εντόπισε ανοιχτή έξοδο κινδύνου προς το Boat Deck. Επιβιβάστηκε στη Λέμβο 15 ακριβώς πριν αυτή κατέβει στη θάλασσα."
    },
    "👶 Master Michel Marcel Navratil (2η Θέση - 3 ετών, 'Ορφανά Τιτανικού', Λέμβος D) [ID #194]": {
        "name": "Master Michel Marcel Navratil", "age": 3, "sex": "male", "pclass": 2, "title": "Master",
        "nationality": "French", "embarked": "Southampton (S)", "companions": "Small Group (3-4 άτομα - Οικογένεια)", "deck": "E", "side": "Port (Αριστερά - Lightoller)",
        "fare": 26.0, "dataset_id": 194, "ticket": "230080", "actual_survived": 1, "boat": "Πτυσσόμενη D (Port)",
        "insight": "3χρονο αγόρι. Ο πατέρας του Michel Navratil Sr. (#149) απήγαγε αυτόν και τον 2χρονο αδελφό του Edmond (#341) από τη μητέρα τους στη Γαλλία και ταξίδευε με ψευδώνυμο ('Louis Hoffman') για τη Νέα Υόρκη. Τη νύχτα του ναυαγίου, ο πατέρας κατάφερε να περάσει τα δύο νήπια στην τελευταία επιτυχημένη λέμβο (Πτυσσόμενη D), λέγοντας στον Michel: «Πες στη μητέρα σου ότι την αγαπούσα». Ο πατέρας πνίγηκε και τα αγόρια έγιναν διάσημα παγκοσμίως ως 'Τα Ορφανά του Τιτανικού' μέχρι να εντοπιστεί η μητέρα τους στη Νίκαια."
    },
    "💔 Master Harold Victor Goodwin (3η Θέση - 9 ετών, Οικογένεια 8 ατόμων) [ID #481]": {
        "name": "Master Harold Victor Goodwin", "age": 9, "sex": "male", "pclass": 3, "title": "Master",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Large Group (5+ άτομα - Μεγάλη οικογένεια)", "deck": "G", "side": "Port (Αριστερά - Lightoller)",
        "fare": 46.90, "dataset_id": 481, "ticket": "CA 2144", "actual_survived": 0, "boat": "Καμία (Χάθηκε στο ναυάγιο)",
        "insight": "9χρονο αγόρι από το Fulham της Αγγλίας. Ταξίδευε με τους γονείς του και τα 5 αδέλφια του (οικογένεια 8 ατόμων). Λόγω του τεράστιου μεγέθους της οικογένειας (Ticket Group 5+), της θέσης τους βαθιά στο κύτος της 3ης θέσης και του αποκλεισμού των μεταναστών, κανένα από τα 8 μέλη της οικογένειας Goodwin δεν επιβίωσε. Το 19 μηνών αδελφάκι του, Sidney (#387), ανασύρθηκε από το πέλαγος και ταυτοποιήθηκε το 2007 μέσω DNA ως 'Το Άγνωστο Παιδί'."
    },
    "🇬🇷 Απόστολος Χρονόπουλος (3η Θέση - 26 ετών, Μετανάστης από Άγιο Σώστη Μεσσηνίας) [ID #74]": {
        "name": "Απόστολος Χρονόπουλος (Apostolos Chronopoulos)", "age": 26, "sex": "male", "pclass": 3, "title": "Mr",
        "nationality": "Greek / Balkan", "embarked": "Cherbourg (C)", "companions": "Pair (2 άτομα - π.χ. με σύντροφο/φίλο)", "deck": "G", "side": "Port (Αριστερά - Lightoller)",
        "fare": 14.45, "dataset_id": 74, "ticket": "2680", "actual_survived": 0, "boat": "Καμία (Χάθηκε στο ναυάγιο)",
        "insight": "26χρονος Έλληνας εργάτης από τον Άγιο Σώστη Μεσσηνίας. Επιβιβάστηκε στο Χερβούργο (Cherbourg) με τον 19χρονο αδελφό του Δημήτριο και εισιτήριο #2680 για να αναζητήσουν μια καλύτερη ζωή στις ΗΠΑ. Και οι 4 Έλληνες επιβάτες του Τιτανικού (όλοι νέοι από το ίδιο χωριό) χάθηκαν τη νύχτα του ναυαγίου, εγκλωβισμένοι στα κατώτερα καταστρώματα χωρίς να κατανοούν τις αγγλόφωνες εντολές του πληρώματος."
    }
}

def apply_preset(preset_key):
    if preset_key in PRESETS and PRESETS[preset_key] is not None:
        p = PRESETS[preset_key]
        st.session_state.last_preset = preset_key
        st.session_state.p_name = p["name"]
        st.session_state.p_age = p["age"]
        st.session_state.p_sex = p["sex"]
        st.session_state.p_pclass = p["pclass"]
        st.session_state.p_title = p["title"]
        st.session_state.p_nationality = p["nationality"]
        st.session_state.p_embarked = p["embarked"]
        st.session_state.p_companions = p["companions"]
        st.session_state.p_deck = p["deck"]
        st.session_state.p_side = p["side"]
        st.session_state.p_fare = p.get("fare", 85.0 if p["pclass"] == 1 else (13.0 if p["pclass"] == 2 else 7.5))
        st.session_state.p_dataset_id = p.get("dataset_id", None)
        st.session_state.p_ticket = p.get("ticket", "")
        st.session_state.p_actual_survived = p.get("actual_survived", None)
        st.session_state.p_insight = p.get("insight", "")
        st.session_state.p_boat = p.get("boat", "")

def render_passenger_banner(screen_step):
    p_name = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "Επιβάτης Τιτανικού"
    is_dataset_preset = st.session_state.get("p_dataset_id") is not None
    
    col_pinfo, col_pswitch = st.columns([3, 2], gap="medium")
    
    with col_pswitch:
        preset_keys = list(PRESETS.keys())
        cur_idx = preset_keys.index(st.session_state.last_preset) if st.session_state.last_preset in preset_keys else 0
        new_preset = st.selectbox(
            "🎭 Αλλαγή Επιβάτη (Preset):",
            preset_keys,
            index=cur_idx,
            key=f"banner_preset_switch_{screen_step}",
            help="Μπορείτε να αλλάξετε επιβάτη ανά πάσα στιγμή από οποιαδήποτε οθόνη."
        )
        if new_preset != st.session_state.last_preset:
            apply_preset(new_preset)
            st.rerun()

    with col_pinfo:
        badge_outcome = ""
        if is_dataset_preset:
            if st.session_state.p_actual_survived == 1:
                badge_outcome = f'<span style="background: rgba(82, 183, 136, 0.25); color: #52b788; border: 1px solid #52b788; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">✅ ΕΠΕΖΗΣΕ ({st.session_state.p_boat})</span>'
            else:
                badge_outcome = '<span style="background: rgba(230, 57, 70, 0.25); color: #e63946; border: 1px solid #e63946; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">❌ ΧΑΘΗΚΕ ΣΤΟ ΝΑΥΑΓΙΟ</span>'

        dataset_id_str = f" • <span style='color: #ffd166; font-weight: 700;'>Kaggle ID #{st.session_state.p_dataset_id}</span>" if is_dataset_preset else ""
        ticket_str = f" • Εισιτήριο: <b>{st.session_state.p_ticket}</b>" if st.session_state.get("p_ticket") else ""
        fare_val = float(st.session_state.get("p_fare", 13.0))
        side_short = "Δεξιά (Starboard)" if "Starboard" in st.session_state.p_side else "Αριστερά (Port)"

        render_html(f"""
        <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid rgba(212, 175, 55, 0.35); border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 1.12rem; font-weight: 700; color: #ffffff;">
                    👤 {p_name} <span style="font-size: 0.88rem; font-weight: normal; color: #a4b8d1;">({st.session_state.p_pclass}η Θέση • {st.session_state.p_age} ετών • {st.session_state.p_sex.capitalize()}{dataset_id_str})</span>
                </div>
                <div>{badge_outcome}</div>
            </div>
            <div style="font-size: 0.84rem; color: #a0c4e2; margin-top: 6px;">
                📍 <b>Κατάστρωμα:</b> Deck {st.session_state.p_deck} • <b>Πλευρά:</b> {side_short} • <b>Ναύλος:</b> £{fare_val:.2f}{ticket_str} • <b>Λιμάνι:</b> {st.session_state.p_embarked.split()[0]}
            </div>
        </div>
        """)

    # Εμφάνιση Κάρτας Insights & Πραγματικών Δεδομένων από το Dataset σε ΟΛΕΣ τις οθόνες
    if is_dataset_preset and st.session_state.get("p_insight"):
        render_html(f"""
        <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid rgba(212, 175, 55, 0.4); border-radius: 12px; padding: 16px 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <div style="font-family: 'Cinzel', serif; font-size: 1.05rem; color: #ffd166; font-weight: 700; margin-bottom: 8px;">
                📋 Ιστορικό Προφίλ & Insights Dataset: {p_name} (#{st.session_state.p_dataset_id})
            </div>
            <div style="font-size: 0.92rem; line-height: 1.6; color: #e2e8f0; background: rgba(0,0,0,0.3); padding: 12px 16px; border-radius: 8px; border-left: 4px solid #d4af37;">
                {st.session_state.p_insight}
            </div>
        </div>
        """)

df_boats = pd.DataFrame(LIFEBOATS)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT FOR WIZARD / STEPPER
# ─────────────────────────────────────────────────────────────────────────────
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

# Passenger profile state
if "p_name" not in st.session_state:
    st.session_state.p_name = ""
if "p_age" not in st.session_state:
    st.session_state.p_age = 28
if "p_sex" not in st.session_state:
    st.session_state.p_sex = "male"
if "p_pclass" not in st.session_state:
    st.session_state.p_pclass = 2
if "p_title" not in st.session_state:
    st.session_state.p_title = "Mr"
if "p_nationality" not in st.session_state:
    st.session_state.p_nationality = "British (UK)"
if "p_embarked" not in st.session_state:
    st.session_state.p_embarked = "Southampton (S)"
if "p_companions" not in st.session_state:
    st.session_state.p_companions = "Solo (Μόνος/η)"
if "p_deck" not in st.session_state:
    st.session_state.p_deck = "D"
if "p_side" not in st.session_state:
    st.session_state.p_side = "Starboard (Δεξιά - Murdoch)"
if "p_fare" not in st.session_state:
    st.session_state.p_fare = 13.0
if "p_dataset_id" not in st.session_state:
    st.session_state.p_dataset_id = None
if "p_ticket" not in st.session_state:
    st.session_state.p_ticket = ""
if "p_actual_survived" not in st.session_state:
    st.session_state.p_actual_survived = None
if "p_insight" not in st.session_state:
    st.session_state.p_insight = ""
if "p_boat" not in st.session_state:
    st.session_state.p_boat = ""
if "last_preset" not in st.session_state:
    st.session_state.last_preset = "⚙️ Προσαρμοσμένος Επιβάτης (Custom Passenger)"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & APP TITLE
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="color: #ffd166; margin-bottom: 5px; font-size: 2.3rem;">R.M.S. TITANIC</h1>
    <h3 style="color: #a0c4e2; font-weight: 300; font-size: 1.2rem; letter-spacing: 2px;">
        ΔΙΑΔΡΑΣΤΙΚΗ ΕΜΠΕΙΡΙΑ ΠΡΟΣΟΜΟΙΩΣΗΣ, 3D ΜΟΝΤΕΛΟ & ΜΗΧΑΝΙΚΗ ΜΑΘΗΣΗ
    </h3>
    <p style="color: #8da2ba; max-width: 800px; margin: 0 auto; font-size: 0.95rem;">
        Ζήστε το ιστορικό ταξίδι της 14ης Απριλίου 1912. Επιλέξτε το προφίλ σας, εντοπίστε την καμπίνα σας στον 3D Τιτανικό, 
        παρακολουθήστε την προσομοίωση του ναυαγίου και ανακαλύψτε τις πραγματικές πιθανότητες επιβίωσής σας.
    </p>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE STEPPER NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
steps = [
    ("1. 🎟️ Εισιτήριο & Ταυτότητα", 1),
    ("2. 🚢 3D Θέση στο Πλοίο", 2),
    ("3. ⏱️ Η Νύχτα του Ναυαγίου", 3),
    ("4. 🔮 Πιθανότητα Επιβίωσης", 4),
    ("5. 📊 Key Notes & Στατιστικά", 5),
]

cols_nav = st.columns(5)
for i, (label, s_idx) in enumerate(steps):
    with cols_nav[i]:
        if st.session_state.current_step == s_idx:
            st.button(f"● {label}", key=f"nav_{s_idx}", use_container_width=True, type="primary")
        else:
            if st.button(label, key=f"nav_{s_idx}", use_container_width=True):
                st.session_state.current_step = s_idx
                st.rerun()

render_html("<hr style='border: 0; border-top: 1px solid rgba(212,175,55,0.25); margin: 20px 0 30px 0;'>")

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 1: BOARDING PASS & 1912 PERSONA SELECTION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.current_step == 1:
    st.markdown("### 🎟️ Βήμα 1: Έκδοση Εισιτηρίου & Ταυτότητα Επιβάτη (1912)")
    st.info("💡 Επιλέξτε ένα έτοιμο ιστορικό πρότυπο (Preset) ή ρυθμίστε χειροκίνητα τα χαρακτηριστικά σας για να επιβιβαστείτε στον Τιτανικό.")

    # Archetype selector: ΜΟΝΟ πραγματικά ιστορικά πρόσωπα από τα αρχεία του Τιτανικού & Kaggle Dataset
    presets = PRESETS
    cur_p_idx = list(presets.keys()).index(st.session_state.last_preset) if st.session_state.last_preset in presets else 0
    selected_preset = st.selectbox("🎭 Επιλέξτε Πραγματικό Ιστορικό Επιβάτη (ή φτιάξτε δικό σας):", list(presets.keys()), index=cur_p_idx)
    if selected_preset != st.session_state.last_preset:
        apply_preset(selected_preset)
        st.rerun()

    # Εμφάνιση Κάρτας Insights & Πραγματικών Δεδομένων από το Dataset
    if st.session_state.get("p_dataset_id") is not None:
        if st.session_state.p_actual_survived == 1:
            surv_badge = '<span style="background: rgba(82, 183, 136, 0.25); color: #52b788; border: 1px solid #52b788; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">✅ Πραγματική Έκβαση: ΕΠΕΖΗΣΕ (Survived: 1)</span>'
        else:
            surv_badge = '<span style="background: rgba(230, 57, 70, 0.25); color: #e63946; border: 1px solid #e63946; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">❌ Πραγματική Έκβαση: ΧΑΘΗΚΕ (Survived: 0)</span>'
        
        boat_badge = f'<div>🚣 <b>Λέμβος Διάσωσης:</b> <span style="color: #ffd166;">{st.session_state.p_boat}</span></div>' if st.session_state.p_boat else ""

        render_html(f"""
        <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid rgba(212, 175, 55, 0.4); border-radius: 12px; padding: 18px 22px; margin: 15px 0 25px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <div style="font-family: 'Cinzel', serif; font-size: 1.15rem; color: #ffd166; font-weight: 700; letter-spacing: 1px;">
                    📋 Στοιχεία Πραγματικού Επιβάτη (Kaggle: yasserh/titanic-dataset)
                </div>
                <div>
                    {surv_badge}
                </div>
            </div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 14px; font-size: 0.88rem; color: #a4b8d1; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px;">
                <div>🎫 <b>Kaggle PassengerId:</b> <span style="color: #ffd166; font-weight: 700;">#{st.session_state.p_dataset_id}</span></div>
                <div>🎟️ <b>Αρ. Εισιτηρίου:</b> <span style="color: #ffffff; font-weight: 600;">{st.session_state.p_ticket}</span></div>
                <div>💰 <b>Πραγματικός Ναύλος:</b> <span style="color: #48cae4; font-weight: 700;">£{st.session_state.p_fare:.2f}</span></div>
                <div>⚓ <b>Λιμάνι:</b> <span style="color: #ffffff;">{st.session_state.p_embarked.split()[0]}</span></div>
                {boat_badge}
            </div>
            <div style="font-size: 0.94rem; line-height: 1.65; color: #e2e8f0; background: rgba(0,0,0,0.3); padding: 14px 18px; border-radius: 8px; border-left: 4px solid #d4af37;">
                {st.session_state.p_insight}
            </div>
        </div>
        """)

    col_form, col_card = st.columns([3, 2], gap="large")

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            name_input = st.text_input("Ονοματεπώνυμο Επιβάτη:", value=st.session_state.p_name, placeholder="π.χ. John Smith ή το όνομά σας")
            st.session_state.p_name = name_input

            sex_opts = ["male", "female"]
            sex_labels = {"male": "👨 Άνδρας / Αγόρι", "female": "👩 Γυναίκα / Κορίτσι"}
            sex_input = st.selectbox("Βιολογικό Φύλο:", sex_opts, index=sex_opts.index(st.session_state.p_sex), format_func=lambda x: sex_labels[x])
            st.session_state.p_sex = sex_input

            # Δυναμικό φιλτράρισμα τίτλων με βάση το επιλεγμένο βιολογικό φύλο
            if sex_input == "male":
                title_opts = ["Mr", "Master", "Rare"]
                title_labels = {
                    "Mr": "Mr. (Ενήλικος άνδρας ≥ 14 ετών)",
                    "Master": "Master. (Ανήλικο αγόρι ≤ 14 ετών)",
                    "Rare": "Τίτλος Ευγενείας / Αξιωματικός (Dr, Col, Rev, Sir)"
                }
            else:
                title_opts = ["Miss", "Mrs", "Rare"]
                title_labels = {
                    "Mrs": "Mrs. (Έγγαμη γυναίκα)",
                    "Miss": "Miss. (Άγαμη γυναίκα / ανήλικο κορίτσι)",
                    "Rare": "Τίτλος Ευγενείας (Lady, Countess, Dona)"
                }

            if st.session_state.p_title not in title_opts:
                st.session_state.p_title = title_opts[0]
            default_t_idx = title_opts.index(st.session_state.p_title)

            title_input = st.selectbox(
                "Κοινωνικός Τίτλος (Title 1912):", 
                title_opts, 
                index=default_t_idx, 
                format_func=lambda x: title_labels[x],
                help="Οι διαθέσιμοι τίτλοι προσαρμόζονται αυτόματα στο επιλεγμένο βιολογικό φύλο."
            )
            st.session_state.p_title = title_input

        with c2:
            # Δυναμικό όριο ηλικίας με βάση τον τίτλο: Αγόρι (Master) αυστηρά 1-14 έτη!
            if title_input == "Master":
                min_age, max_age = 1, 14
                age_help = "Για ανήλικο αγόρι (Master), το όριο ηλικίας είναι αυστηρά έως 14 ετών."
            elif title_input == "Mr":
                min_age, max_age = 14, 80
                age_help = "Για ενήλικο άνδρα (Mr.), η ηλικία είναι από 14 ετών και άνω."
            elif title_input == "Mrs":
                min_age, max_age = 15, 80
                age_help = "Για έγγαμη γυναίκα (Mrs.), η ηλικία είναι από 15 ετών και άνω."
            elif title_input == "Rare":
                min_age, max_age = 18, 80
                age_help = "Για τίτλους ευγενείας ή αξιωματικούς, η ηλικία είναι από 18 ετών και άνω."
            else: # Miss
                min_age, max_age = 1, 80
                age_help = "Για Miss, η ηλικία μπορεί να είναι από 1 έως 80 ετών."

            cur_age_clamped = int(np.clip(st.session_state.p_age, min_age, max_age))
            st.session_state.p_age = cur_age_clamped

            age_input = st.slider(
                "Ηλικία (σε έτη):", 
                min_value=min_age, 
                max_value=max_age, 
                value=cur_age_clamped,
                help=age_help
            )
            st.session_state.p_age = age_input

            pclass_opts = [1, 2, 3]
            pclass_labels = {1: "🥇 1η Θέση (First Class - Deluxe)", 2: "🥈 2η Θέση (Second Class - Cabin)", 3: "🥉 3η Θέση (Third Class - Steerage)"}
            pclass_input = st.selectbox("Κοινωνικοοικονομική Θέση (Class):", pclass_opts, index=pclass_opts.index(st.session_state.p_pclass), format_func=lambda x: pclass_labels[x])
            st.session_state.p_pclass = pclass_input

            nat_opts = [
                "British (UK)", "Irish (Ιρλανδία)", "American (USA)", 
                "Scandinavian (Σουηδία/Νορβηγία)", "Greek / Balkan", 
                "Syrian / Lebanese (Οθωμανική Αυτοκρατορία)", "French"
            ]
            nat_default_idx = nat_opts.index(st.session_state.p_nationality) if st.session_state.p_nationality in nat_opts else 0
            nat_input = st.selectbox("Εθνικότητα / Χώρα Καταγωγής:", nat_opts, index=nat_default_idx)
            st.session_state.p_nationality = nat_input

        st.markdown("#### 👥 Συνοδοί & Λιμάνι Επιβίβασης")
        c3, c4 = st.columns(2)
        with c3:
            comp_opts = [
                "Solo (Μόνος/η)",
                "Pair (2 άτομα - π.χ. με σύντροφο/φίλο)",
                "Small Group (3-4 άτομα - Οικογένεια)",
                "Large Group (5+ άτομα - Μεγάλη οικογένεια)"
            ]
            comp_default_idx = comp_opts.index(st.session_state.p_companions) if st.session_state.p_companions in comp_opts else 0
            comp_input = st.selectbox("Μέγεθος Ομάδας Εισιτηρίου (Ticket Group):", comp_opts, index=comp_default_idx)
            st.session_state.p_companions = comp_input

        with c4:
            emb_opts = ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
            emb_labels = {
                "Southampton (S)": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Southampton (Αγγλία - Αφετηρία)",
                "Cherbourg (C)": "🇫🇷 Cherbourg (Γαλλία - Αριστοκράτες)",
                "Queenstown (Q)": "🇮🇪 Queenstown / Cobh (Ιρλανδία - Μετανάστες)"
            }
            emb_default_idx = 0
            for idx, item in enumerate(emb_opts):
                if st.session_state.p_embarked.startswith(item[:3]):
                    emb_default_idx = idx
            emb_input = st.selectbox("Λιμάνι Επιβίβασης:", emb_opts, index=emb_default_idx, format_func=lambda x: emb_labels[x])
            st.session_state.p_embarked = emb_input

    with col_card:
        # Calculate fare in 1912 pounds and modern euros (1 £ in 1912 ≈ €135 today)
        fare_1912 = float(st.session_state.get("p_fare", 85.0 if st.session_state.p_pclass == 1 else (13.0 if st.session_state.p_pclass == 2 else 7.5)))
        fare_modern = int(fare_1912 * 135)
        
        if st.session_state.p_pclass == 1:
            class_str = "FIRST CLASS PARLOUR"
        elif st.session_state.p_pclass == 2:
            class_str = "SECOND CLASS CABIN"
        else:
            class_str = "THIRD CLASS STEERAGE"

        port_short = st.session_state.p_embarked.split()[0]
        origin_short = st.session_state.p_nationality.split()[0]
        sex_capital = st.session_state.p_sex.capitalize()
        pass_name_display = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "Επιβάτης Τιτανικού"
        ticket_str = f" • TICKET: {st.session_state.p_ticket}" if st.session_state.get("p_ticket") else ""

        # Render Boarding Pass safely using render_html
        render_html(f"""
        <div class="boarding-pass">
            <div class="boarding-header">PASSENGER TICKET</div>
            <div class="boarding-sub">OCEANIC STEAM NAVIGATION CO. • R.M.S. TITANIC{ticket_str}</div>
            <div style="margin-bottom: 12px;">
                <div class="pass-label">PASSENGER NAME</div>
                <div class="pass-val">{pass_name_display}</div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <div>
                    <div class="pass-label">CLASS</div>
                    <div style="color: #ffd166; font-weight: 700;">{class_str}</div>
                </div>
                <div>
                    <div class="pass-label">AGE / SEX</div>
                    <div style="color: #ffffff; font-weight: 700;">{st.session_state.p_age} yrs • {sex_capital}</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 14px;">
                <div>
                    <div class="pass-label">EMBARKATION</div>
                    <div style="color: #48cae4; font-weight: 700;">{port_short}</div>
                </div>
                <div>
                    <div class="pass-label">ORIGIN</div>
                    <div style="color: #ffffff; font-weight: 700;">{origin_short}</div>
                </div>
            </div>
            <div style="background: rgba(0,0,0,0.35); border-radius: 8px; padding: 10px; border: 1px dashed rgba(212,175,55,0.4);">
                <div class="pass-label">TICKET FARE & VALUE</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 1.15rem; color: #ffd166; font-weight: 700;">£{fare_1912:.2f} (1912)</span>
                    <span style="font-size: 0.85rem; color: #a4b8d1;">≈ €{fare_modern:,} (Αξία 2026)</span>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 0.72rem; color: #8899aa; text-align: center; letter-spacing: 1px;">
                WHITE STAR LINE • SAILING DATE: APRIL 10, 1912
            </div>
        </div>
        """)

        # Historical context note
        if st.session_state.p_nationality == "Greek / Balkan":
            st.caption("ℹ️ **Ιστορικό Σημείωμα (Έλληνες Επιβάτες)**: Στον Τιτανικό επέβαιναν ακριβώς 4 καταγεγραμμένοι Έλληνες μετανάστες, όλοι από το χωριό Άγιος Σώστης Μεσσηνίας: ο Απόστολος Χρονόπουλος (26 ετών - καταγεγραμμένος ως επιβάτης #74 στο Kaggle dataset), ο αδελφός του Δημήτριος (19 ετών), ο Βασίλειος Κατάβελος (19 ετών) και ο Παναγιώτης Λυμπερόπουλος (30 ετών). Επιβιβάστηκαν όλοι στο Χερβούργο (Cherbourg) στην 3η θέση και δυστυχώς χάθηκαν και οι 4 στο ναυάγιο.")
        elif st.session_state.p_pclass == 3:
            st.caption("ℹ️ **Ιστορικό Σημείωμα 3ης Θέσης**: Οι επιβάτες 3ης θέσης έπρεπε να περάσουν από ιατρικό έλεγχο για τράχωμα και ψείρες πριν επιβιβαστούν, ώστε να μην απορριφθούν από τις αρχές του Ellis Island στις ΗΠΑ.")

    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    _, next_col = st.columns([4, 1])
    with next_col:
        if st.button("Επόμενο: 3D Θέση στο Πλοίο ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 2: 3D SHIP VISUALIZATION & CABIN PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 2:
    st.markdown("### 🚢 Βήμα 2: Προσομοίωση Θέσης & Καμπίνας στον 3D Τιτανικό")
    st.markdown("""
    Δείτε πού βρισκόταν η καμπίνα σας σε τρισδιάστατο περιβάλλον. Η πλευρά του πλοίου (**Αριστερά / Port** vs **Δεξιά / Starboard**) 
    και το κατάστρωμα καθόρισαν σε τεράστιο βαθμό την πρόσβαση στις σωσίβιες λέμβους.
    """)

    # Εμφάνιση του ενεργού επιβάτη & των ιστορικών insights του dataset
    render_passenger_banner(2)

    c_inputs, c_view = st.columns([1, 2], gap="large")

    with c_inputs:
        st.markdown("#### ⚓ Επιλογή Καταστρώματος & Πλευράς")
        
        # Valid decks by class
        if st.session_state.p_pclass == 1:
            deck_options = ["A", "B", "C"]
            deck_help = "Καταστρώματα A-C: Πολυτελείς σουίτες, άμεση πρόσβαση στο Boat Deck και τη Μεγάλη Σκάλα."
        elif st.session_state.p_pclass == 2:
            deck_options = ["D", "E"]
            deck_help = "Καταστρώματα D-E: Μεσαίο ύψος, τραπεζαρία 2ης θέσης, 3 καταστρώματα κάτω από τις λέμβους."
        else:
            deck_options = ["E", "F", "G"]
            deck_help = "Καταστρώματα E-G: Κάτω από την ίσαλο γραμμή ή στο κύτος. Λαβύρινθος από διαδρόμους και στεγανές θύρες."

        current_deck = st.session_state.p_deck if st.session_state.p_deck in deck_options else deck_options[0]
        selected_deck = st.selectbox(f"Κατάστρωμα (Deck) για {st.session_state.p_pclass}η Θέση:", deck_options, index=deck_options.index(current_deck), help=deck_help)
        st.session_state.p_deck = selected_deck

        side_options = [
            "Port (Αριστερά - Lightoller)", 
            "Starboard (Δεξιά - Murdoch)"
        ]
        side_idx = 1 if "Starboard" in st.session_state.p_side else 0
        selected_side = st.radio("Πλευρά του Πλοίου (Ship Side):", side_options, index=side_idx)
        st.session_state.p_side = selected_side

        is_starboard = "Starboard" in selected_side

        # Distance & escape analysis
        deck_dist_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        flights = deck_dist_map.get(selected_deck, 4)
        meters_to_boats = flights * 3.2
        officer_name = 'William Murdoch' if is_starboard else 'Charles Lightoller'
        altitude = 18 - flights * 2.2

        render_html(f"""
        <div class="info-card">
            <div style="font-weight: 700; color: #ffd166; margin-bottom: 8px;">📍 Στοιχεία Τοποθεσίας Καμπίνας</div>
            <div>• <b>Επιβάτης:</b> <span style="color: #ffffff; font-weight: 700;">{st.session_state.p_name}</span> ({st.session_state.p_pclass}η Θέση, {st.session_state.p_age} ετών)</div>
            <div>• <b>Κατάστρωμα:</b> Deck {selected_deck}</div>
            <div>• <b>Υψόμετρο από Ίσαλο:</b> {altitude:.1f} μέτρα</div>
            <div>• <b>Σκάλες μέχρι το Boat Deck:</b> {flights} επίπεδα ({meters_to_boats:.1f} m)</div>
            <div>• <b>Αξιωματικός Υπεύθυνος:</b> <span class="highlight-cyan">{officer_name}</span></div>
        </div>
        """)

        is_woman_or_child = (st.session_state.p_sex == "female") or (st.session_state.p_age < 14)

        if not is_starboard:
            if is_woman_or_child:
                st.success("✅ **Port Side (Αριστερά - Lightoller)**: Εξαιρετικά ευνοϊκή θέση για εσάς! Ο Υποπλοίαρχος Lightoller εφάρμοσε αυστηρά το *'Γυναίκες και παιδιά ΜΟΝΟΝ'*. Σας δίνεται απόλυτη προτεραιότητα επιβίβασης στις σωσίβιες λέμβους έναντι όλων των ανδρών.")
            else:
                st.error("⚠️ **Port Side (Αριστερά - Lightoller)**: Εξαιρετικά επικίνδυνη πλευρά για ενήλικο άνδρα! Ο Lightoller εφάρμοσε το *'Γυναίκες και παιδιά ΜΟΝΟΝ'* και αρνούνταν την επιβίβαση σε άνδρες ακόμα κι όταν οι λέμβοι έφευγαν με δεκάδες κενές θέσεις.")
        else:
            if is_woman_or_child:
                st.info("ℹ️ **Starboard Side (Δεξιά - Murdoch)**: Ο Υποπλοίαρχος Murdoch εφάρμοσε το *'Γυναίκες και παιδιά ΠΡΩΤΑ'*. Έχετε άμεση προτεραιότητα, αλλά επιτρέπεται και σε άνδρες να επιβιβαστούν αν υπάρχουν κενά καθίσματα.")
            else:
                st.success("✅ **Starboard Side (Δεξιά - Murdoch)**: Η πιο ευνοϊκή πλευρά για άνδρα επιβάτη! Ο Murdoch εφάρμοσε το *'Γυναίκες και παιδιά ΠΡΩΤΑ'*. Αν δεν υπήρχαν γυναίκες στην ουρά, επέτρεπε σε άνδρες να επιβιβαστούν, αυξάνοντας σημαντικά τις πιθανότητες σωτηρίας σας.")

    with c_view:
        # Camera angle view selector
        cam_choice = st.radio(
            "🎥 Οπτική Γωνία 3D Κάμερας:",
            ["Ισομετρική 3D (Default)", "Πλάγια Starboard (Δεξιά)", "Πλάγια Port (Αριστερά)", "Κάτοψη (Top)", "Πλώρη (Bow)"],
            horizontal=True
        )

        cam_map = {
            "Ισομετρική 3D (Default)": dict(eye=dict(x=2.0, y=1.8, z=1.2), center=dict(x=0, y=0, z=-0.05)),
            "Πλάγια Starboard (Δεξιά)": dict(eye=dict(x=0.05, y=3.1, z=0.45), center=dict(x=0, y=0, z=-0.05)),
            "Πλάγια Port (Αριστερά)": dict(eye=dict(x=0.05, y=-3.1, z=0.45), center=dict(x=0, y=0, z=-0.05)),
            "Κάτοψη (Top)": dict(eye=dict(x=0.0, y=0.0, z=3.4), center=dict(x=0, y=0, z=0), up=dict(x=1, y=0, z=0)),
            "Πλώρη (Bow)": dict(eye=dict(x=2.9, y=0.1, z=0.6), center=dict(x=0, y=0, z=-0.05))
        }
        current_cam = cam_map.get(cam_choice, cam_map["Ισομετρική 3D (Default)"])

        # High-Fidelity 3D Titanic Model
        fig3d = go.Figure()

        # 1. Hull Parameters (Multi-station parametric curved hull)
        x_hull = np.linspace(0, 100, 35)
        def hw(x):
            if x < 18:
                return 13.0 * np.sin((x / 18.0) * (np.pi / 2))
            elif x <= 72:
                return 13.0
            else:
                return 13.0 * (((100.0 - x) / 28.0) ** 0.55)

        # Upper Hull (Midnight Naval Black)
        z_hull_u = np.linspace(3.5, 12.0, 5)
        X_u, Z_u = np.meshgrid(x_hull, z_hull_u)
        Y_u_stbd = np.zeros_like(X_u)
        for i, x in enumerate(x_hull):
            Y_u_stbd[:, i] = hw(x)

        fig3d.add_trace(go.Surface(
            x=X_u, y=Y_u_stbd, z=Z_u,
            colorscale=[[0, '#111720'], [1, '#111720']],
            showscale=False, opacity=0.96, name='Κύτος (Starboard)', hoverinfo='skip'
        ))
        fig3d.add_trace(go.Surface(
            x=X_u, y=-Y_u_stbd, z=Z_u,
            colorscale=[[0, '#111720'], [1, '#111720']],
            showscale=False, opacity=0.96, name='Κύτος (Port)', hoverinfo='skip'
        ))

        # Lower Keel / Underbody (Red Oxide Boot-topping)
        z_hull_l = np.linspace(0.0, 3.5, 4)
        X_l, Z_l = np.meshgrid(x_hull, z_hull_l)
        Y_l_stbd = np.zeros_like(X_l)
        frac = z_hull_l / 3.5
        for i, x in enumerate(x_hull):
            Y_l_stbd[:, i] = hw(x) * (0.65 + 0.35 * frac)

        fig3d.add_trace(go.Surface(
            x=X_l, y=Y_l_stbd, z=Z_l,
            colorscale=[[0, '#781818'], [1, '#781818']],
            showscale=False, opacity=0.96, name='Ίσαλος / Καρίνα', hoverinfo='skip'
        ))
        fig3d.add_trace(go.Surface(
            x=X_l, y=-Y_l_stbd, z=Z_l,
            colorscale=[[0, '#781818'], [1, '#781818']],
            showscale=False, opacity=0.96, name='Ίσαλος / Καρίνα', hoverinfo='skip'
        ))

        # Superstructure Walls (White Deckhouse, x 18 to 72)
        x_sup = np.linspace(18, 72, 20)
        z_sup = np.linspace(12.0, 18.5, 4)
        X_s, Z_s = np.meshgrid(x_sup, z_sup)
        Y_s = np.full_like(X_s, 7.8)

        fig3d.add_trace(go.Surface(
            x=X_s, y=Y_s, z=Z_s,
            colorscale=[[0, '#edf2f7'], [1, '#edf2f7']],
            showscale=False, opacity=0.88, name='Υπερκατασκευή', hoverinfo='skip'
        ))
        fig3d.add_trace(go.Surface(
            x=X_s, y=-Y_s, z=Z_s,
            colorscale=[[0, '#edf2f7'], [1, '#edf2f7']],
            showscale=False, opacity=0.88, name='Υπερκατασκευή', hoverinfo='skip'
        ))

        # Teak Wood Boat Deck (Top, z 18.5, x 18 to 72, y -7.8 to 7.8)
        xd = np.linspace(18, 72, 15)
        yd = np.linspace(-7.8, 7.8, 8)
        XD, YD = np.meshgrid(xd, yd)
        fig3d.add_trace(go.Surface(
            x=XD, y=YD, z=np.full_like(XD, 18.5),
            colorscale=[[0, '#c2a67e'], [1, '#c2a67e']],
            showscale=False, opacity=0.92, name='Boat Deck (Ξύλινο Κατάστρωμα)', hoverinfo='skip'
        ))

        # Navigating Bridge (x 72, wings extending to y +- 10.5)
        fig3d.add_trace(go.Scatter3d(
            x=[72, 72, 72, 72, 72], y=[-10.5, -7.8, 0, 7.8, 10.5], z=[20.2, 20.2, 20.8, 20.2, 20.2],
            mode='lines+markers', line=dict(color='#ffffff', width=5),
            marker=dict(size=4, color='#ffd166'), name='Γέφυρα Πλοίου (Bridge)'
        ))

        # 4 Iconic Raked Funnels (with 10-degree backward tilt)
        for f_idx, fc_x in enumerate([63, 51, 39, 27]):
            theta = np.linspace(0, 2*np.pi, 20)
            h_vals = np.linspace(0, 11, 6)
            Th, H = np.meshgrid(theta, h_vals)
            X_fun = (fc_x - H * 0.16) + 2.1 * np.cos(Th)
            Y_fun = 2.1 * np.sin(Th)
            Z_fun = 18.5 + H
            fig3d.add_trace(go.Surface(
                x=X_fun, y=Y_fun, z=Z_fun,
                colorscale=[[0, '#c87d20'], [0.78, '#c87d20'], [0.78, '#141414'], [1, '#141414']],
                showscale=False, opacity=0.98,
                name=f'Καπνοδόχος {f_idx+1}', hoverinfo='skip'
            ))

        # Fore and Aft Masts with Crow's Nest
        fig3d.add_trace(go.Scatter3d(
            x=[86, 83], y=[0, 0], z=[12, 36],
            mode='lines', line=dict(color='#d0dbe8', width=3), name='Πλωραίο Κατάρτι'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[84.5], y=[0], z=[27],
            mode='markers+text', marker=dict(size=6, color='#ffd166', symbol='diamond'),
            text=["Παρατηρητήριο (Crow's Nest)"], textposition='top right', name='Παρατηρητήριο'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[10, 8], y=[0, 0], z=[12, 33],
            mode='lines', line=dict(color='#d0dbe8', width=3), name='Πρυμναίο Κατάρτι'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[83, 8], y=[0, 0], z=[36, 33],
            mode='lines', line=dict(color='#ffd166', width=1.5, dash='dash'), name='Κεραία Ασυρμάτου (Marconi)'
        ))

        # Lifeboats along Boat Deck edge
        lb_x = [68, 62, 56, 50, 44, 38, 32, 26]
        lb_y_stbd = [8.5] * len(lb_x)
        lb_y_port = [-8.5] * len(lb_x)
        lb_z = [19.2] * len(lb_x)

        fig3d.add_trace(go.Scatter3d(
            x=lb_x, y=lb_y_stbd, z=lb_z,
            mode='markers', marker=dict(size=6, color='#52b788', symbol='diamond'),
            name='Σωσίβιες Λέμβοι Starboard (Murdoch)'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=lb_x, y=lb_y_port, z=lb_z,
            mode='markers', marker=dict(size=6, color='#48cae4', symbol='diamond'),
            name='Σωσίβιες Λέμβοι Port (Lightoller)'
        ))

        # Ocean Waterline
        x_sea = [-5, 105, 105, -5]
        y_sea = [-25, -25, 25, 25]
        fig3d.add_trace(go.Mesh3d(
            x=x_sea, y=y_sea, z=[3.5, 3.5, 3.5, 3.5],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='#071829', opacity=0.45, name='Ωκεανός (Ίσαλος)', hoverinfo='skip'
        ))

        # USER CABIN LOCATION BEACON
        deck_z_map = {'A': 17.5, 'B': 16.0, 'C': 14.5, 'D': 12.5, 'E': 10.5, 'F': 8.0, 'G': 5.5}
        user_z = deck_z_map.get(selected_deck, 12.0)
        user_y = 5.2 if is_starboard else -5.2
        user_x = 58.0 if st.session_state.p_pclass == 1 else (42.0 if st.session_state.p_pclass == 2 else 24.0)

        # Vertical glowing beacon ray
        fig3d.add_trace(go.Scatter3d(
            x=[user_x, user_x], y=[user_y, user_y], z=[user_z, 35],
            mode='lines', line=dict(color='#ffd166', width=4, dash='dot'),
            name='Ακτίνα Εντοπισμού Καμπίνας'
        ))
        # Glowing marker
        display_label = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "ΕΣΕΙΣ"
        fig3d.add_trace(go.Scatter3d(
            x=[user_x], y=[user_y], z=[user_z],
            mode='markers+text',
            marker=dict(size=14, color='#ffd166', line=dict(color='#ffffff', width=2), symbol='circle'),
            text=[f'📍 Η ΚΑΜΠΙΝΑ ΣΑΣ ({display_label})'],
            textposition='top center',
            textfont=dict(color='#ffd166', size=13),
            name='Η Καμπίνα σας'
        ))

        fig3d.update_layout(
            scene=dict(
                bgcolor='#08101a',
                xaxis=dict(title='← Πρύμνη (0) | Μήκος Πλοίου | Πλώρη (100) →', showgrid=True, gridcolor='#182638', zeroline=False),
                yaxis=dict(title='Port ← | → Starboard', showgrid=True, gridcolor='#182638', zeroline=False),
                zaxis=dict(title='Ύψος (Καταστρώματα)', showgrid=True, gridcolor='#182638', zeroline=False),
                aspectratio=dict(x=3.5, y=1.3, z=0.9),
                camera=current_cam
            ),
            margin=dict(l=0, r=0, t=10, b=10),
            height=540,
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(10,20,35,0.85)', font=dict(color='#ffffff', size=10))
        )

        st.plotly_chart(fig3d, use_container_width=True)

    # Navigation buttons
    col_prev, _, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("⬅️ Πίσω: Εισιτήριο", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col_next:
        if st.button("Επόμενο: Προσομοίωση Ναυαγίου ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 3: TIMELINE SIMULATION ("ΣΑΝ ΒΙΝΤΕΟ")
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 3:
    st.markdown("### ⏱️ Βήμα 3: Η Νύχτα της 14ης Απριλίου 1912 - Χρονολόγιο & Προσομοίωση")
    st.markdown("""
    Παρακολουθήστε την εξέλιξη της βύθισης λεπτό προς λεπτό. Μετακινήστε τον διαδραστικό χρονοδιακόπτη 
    για να δείτε πώς εξελισσόταν η κατάσταση στην καμπίνα σας και στις σωσίβιες λέμβους.
    """)

    # Εμφάνιση του ενεργού επιβάτη & των ιστορικών insights του dataset
    render_passenger_banner(3)

    timeline_events = [
        {
            "time": "23:40",
            "title": "🧊 Σύγκρουση με το Παγόβουνο (Starboard Bow)",
            "desc": "Ο παρατηρητής Frederick Fleet χτυπά την καμπάνα 3 φορές: 'Iceberg right ahead!'. Ο Τιτανικός ξύνει το παγόβουνο στη δεξιά πλευρά (Starboard). Σχίζονται 6 στεγανά διαμερίσματα. Εισροή νερού: 7 τόνοι το δευτερόλεπτο.",
            "water_level": "Κύτος / Διαμερίσματα λεβήτων 5 & 6",
            "active_decks": "Tank Top (Κύτος)",
            "urgency": "Ήρεμη (Οι περισσότεροι επιβάτες δεν ένιωσαν παρά έναν ελαφρύ κραδασμό)",
            "cabins_flooded": ["Tank Top"]
        },
        {
            "time": "00:05",
            "title": "📋 Εντολή Εκκένωσης & Προετοιμασία Λέμβων",
            "desc": "Ο ναυπηγός Thomas Andrews υπολογίζει ότι το πλοίο έχει το πολύ 1.5 - 2 ώρες ζωής. Ο Καπετάνιος Smith διατάζει αποκάλυψη των 20 σωσιβίων λέμβων και εκπομπή σήματος κινδύνου CQD / SOS. Η συνολική χωρητικότητα των λέμβων είναι 1.178 άτομα για 2.224 επιβαίνοντες.",
            "water_level": "Κατάστρωμα G & Ταχυδρομείο",
            "active_decks": "G Deck",
            "urgency": "Μέτρια (Ανακοινώσεις στα σαλόνια 1ης & 2ης θέσης, σύγχυση στην 3η θέση)",
            "cabins_flooded": ["Tank Top", "G"]
        },
        {
            "time": "00:45",
            "title": "🛶 Καθέλκυση Πρώτης Λέμβου (No. 7 - Starboard)",
            "desc": "Ο Υποπλοίαρχος Murdoch καθελκύει τη Λέμβο 7 στη δεξιά πλευρά με μόλις 28 επιβάτες (χωρητικότητα 65!). Οι επιβάτες διστάζουν να εγκαταλείψουν το φωτισμένο θερμαινόμενο πλοίο για τον παγωμένο ωκεανό. Στην αριστερή πλευρά (Port), ο Lightoller ξεκινά να επιβιβάζει αυστηρά μόνο γυναίκες και παιδιά.",
            "water_level": "Κατάστρωμα F (Καμπίνες 3ης θέσης)",
            "active_decks": "F Deck",
            "urgency": "Υψηλή (Εκτόξευση λευκών φωτοβολίδων κινδύνου στον νυχτερινό ουρανό)",
            "cabins_flooded": ["Tank Top", "G", "F"]
        },
        {
            "time": "01:15",
            "title": "🚪 Πλημμύρα στα Κάτω Καταστρώματα & Παγίδευση 3ης Θέσης",
            "desc": "Το νερό κατακλύζει το Κατάστρωμα E. Οι επιβάτες 3ης θέσης συναντούν κλειδωμένες σιδερένιες θύρες και λαβυρινθώδεις διαδρόμους χωρίς σήμανση στη γλώσσα τους. Μόνο λίγοι καταφέρνουν να βρουν έξοδο κινδύνου προς τα επάνω καταστρώματα.",
            "water_level": "Κατάστρωμα E & Σκάλες",
            "active_decks": "E Deck",
            "urgency": "Κρίσιμη (Κλίση της πλώρης προς τα εμπρός, τα φώτα αρχίζουν να τρεμοπαίζουν)",
            "cabins_flooded": ["Tank Top", "G", "F", "E"]
        },
        {
            "time": "01:40",
            "title": "⚡ Πανικός στο Boat Deck & Διαχωρισμός Οικογενειών",
            "desc": "Η πλώρη βυθίζεται ραγδαία. Καθελκύεται η πτυσσόμενη Λέμβος C με 71 άτομα (στην οποία επιβιβάζεται και ο Bruce Ismay). Στην αριστερή πλευρά (Port), ο Lightoller απειλεί με περίστροφο άνδρες που προσπαθούν να πλησιάσουν τις λέμβους. Στη δεξιά (Starboard), ο Murdoch γεμίζει τις τελευταίες λέμβους.",
            "water_level": "Κατάστρωμα D & C",
            "active_decks": "D & C Decks",
            "urgency": "Απόλυτος Πανικός (Οι λέμβοι τελειώνουν, μένουν πάνω από 1.500 άνθρωποι στο πλοίο)",
            "cabins_flooded": ["Tank Top", "G", "F", "E", "D", "C"]
        },
        {
            "time": "02:05",
            "title": "🎻 Τελευταία Πτυσσόμενη Λέμβος & 'Nearer, My God, to Thee'",
            "desc": "Η τελευταία λέμβος (Collapsible D) φεύγει με 44 άτομα. Η ορχήστρα του Wallace Hartley παίζει μουσική μέχρι τέλους στο κατάστρωμα. Ο καπετάνιος Smith ανακοινώνει: 'Every man for himself'.",
            "water_level": "Boat Deck (Κατάστρωμα Λέμβων)",
            "active_decks": "A Deck & Boat Deck",
            "urgency": "Αναπόφευκτο Τέλος",
            "cabins_flooded": ["Tank Top", "G", "F", "E", "D", "C", "B", "A"]
        },
        {
            "time": "02:20",
            "title": "💔 Θραύση του Σκάφους & Τελική Βύθιση",
            "desc": "Η τεράστια τάση σπάει τον Τιτανικό στα δύο μεταξύ 3ου και 4ου καπνοδόχου. Η πλώρη βουλιάζει πρώτη, ακολουθούμενη από την πρύμνη. 1.500 άνθρωποι βρίσκονται στο νερό θερμοκρασίας -2°C. Ο θάνατος από υποθερμία επέρχεται σε 10-15 λεπτά.",
            "water_level": "Πλήρης Βύθιση στον Ατλαντικό (Βάθος 3.800m)",
            "active_decks": "Όλα τα καταστρώματα βυθισμένα",
            "urgency": "Τραγωδία",
            "cabins_flooded": ["Tank Top", "G", "F", "E", "D", "C", "B", "A", "Boat Deck"]
        }
    ]

    # Time slider
    step_time = st.select_slider(
        "🎛️ Επιλέξτε Χρονική Στιγμή (Time of Disaster):",
        options=[ev["time"] for ev in timeline_events],
        value="01:15"
    )

    current_ev = next(ev for ev in timeline_events if ev["time"] == step_time)

    # Check passenger cabin status at this time
    passenger_deck = st.session_state.p_deck
    is_flooded = passenger_deck in current_ev["cabins_flooded"]

    col_tl1, col_tl2 = st.columns([3, 2], gap="large")

    with col_tl1:
        st.markdown(f"### {current_ev['title']}")
        st.markdown(f"<p style='font-size: 1.05rem; line-height: 1.7; color: #d0dbe8;'>{current_ev['desc']}</p>", unsafe_allow_html=True)

        st.markdown("#### 🌊 Κατάσταση Πλημμύρας & Στάθμη Νερού:")
        st.progress(
            (timeline_events.index(current_ev) + 1) / len(timeline_events),
            text=f"Στάθμη Νερού: {current_ev['water_level']}"
        )

        # Status badge for this passenger
        if is_flooded:
            st.error(f"🚨 **Η ΚΑΜΠΙΝΑ ΣΑΣ ΣΤΟ DECK {passenger_deck} ΕΧΕΙ ΠΛΗΜΜΥΡΙΣΕΙ!** Έπρεπε ήδη να έχετε εγκαταλείψει τον χώρο, διαφορετικά είστε παγιδευμένοι.")
        else:
            st.success(f"🛡️ **Η Καμπίνα σας στο Deck {passenger_deck} είναι ακόμα στεγνή**, αλλά ο χρόνος εξαντλείται γρήγορα.")

        if st.session_state.get("p_boat"):
            st.info(f"📍 **Ιστορική Κατάσταση Επιβάτη ({st.session_state.p_name})**: Στα αρχεία του Τιτανικού, ο επιβάτης συνδέεται με τη σωσίβια λέμβο **{st.session_state.p_boat}**.")

    with col_tl2:
        st.markdown("#### 🚣 Διαθέσιμες Λέμβοι ανά Πλευρά")
        
        # Filter boats launched up to this time
        t_minutes = int(current_ev["time"].split(":")[0]) * 60 + int(current_ev["time"].split(":")[1])
        if t_minutes < 12 * 60: # After midnight
            t_minutes += 24 * 60

        def boat_launch_min(t_str):
            h, m = map(int, t_str.split(":"))
            return (h + 24 if h < 12 else h) * 60 + m

        boats_launched = [b for b in LIFEBOATS if boat_launch_min(b["time"]) <= t_minutes]
        p_side_boats = [b for b in boats_launched if ("Starboard" in st.session_state.p_side and b["side"] == "Starboard") or ("Port" in st.session_state.p_side and b["side"] == "Port")]

        st.metric(
            label="Συνολικές Λέμβοι που έχουν αναχωρήσει:",
            value=f"{len(boats_launched)} / 20",
            delta=f"{sum(b['loaded'] for b in boats_launched)} επιζώντες"
        )

        st.markdown(f"**Στη δική σας πλευρά ({'Starboard' if 'Starboard' in st.session_state.p_side else 'Port'}):**")
        if p_side_boats:
            boat_list_str = ", ".join([f"No.{b['boat']} ({b['loaded']}/{b['capacity']} επιβάτες)" for b in p_side_boats[-3:]])
            st.write(f"Τελευταίες λέμβοι: {boat_list_str}")
        else:
            st.write("Καμία λέμβος δεν έχει καθελκυστεί ακόμα από την πλευρά σας.")

        is_woman_or_child = (st.session_state.p_sex == "female") or (st.session_state.p_age < 14)

        if "Starboard" in st.session_state.p_side:
            if is_woman_or_child:
                st.info("💡 **Officer Murdoch (Starboard)**: Έχετε προτεραιότητα επιβίβασης, ενώ επιτρέπεται η συμπλήρωση κενών θέσεων από άνδρες.")
            else:
                st.success("💡 **Officer Murdoch (Starboard)**: Επιτρέπει σε άνδρες να επιβιβαστούν αν δεν περιμένουν γυναίκες, δίνοντάς σας ευκαιρία διαφυγής.")
        else:
            if is_woman_or_child:
                st.success("🛡️ **Officer Lightoller (Port)**: Σας δίνει απόλυτη προτεραιότητα ('Women and children ONLY'), αποκλείοντας όλους τους άνδρες.")
            else:
                st.warning("⚠️ **Officer Lightoller (Port)**: Απαγορεύει αυστηρά την είσοδο σε άνδρες, ακόμα κι αν οι λέμβοι φεύγουν μισοάδειες.")

    # Navigation buttons
    col_prev, _, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("⬅️ Πίσω: 3D Θέση", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col_next:
        if st.button("Επόμενο: Πιθανότητα Επιβίωσης ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 4: SURVIVAL PROBABILITY & MACHINE LEARNING EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 4:
    st.markdown("### 🔮 Βήμα 4: Η Μοίρα σας - Πρόβλεψη Επιβίωσης & ML Επεξήγηση")
    st.markdown("Υπολογισμός πιθανότητας με βάση τα εκπαιδευμένα μοντέλα Μηχανικής Μάθησης και τις ιστορικές συνθήκες της θέσης σας.")

    # Εμφάνιση του ενεργού επιβάτη & των ιστορικών insights του dataset
    render_passenger_banner(4)

    # 1. Prepare feature vector matching training columns
    input_dict = {col: 0.0 for col in feature_cols}
    
    # Numerical features
    input_dict["Age"] = float(st.session_state.p_age)
    input_dict["Fare"] = float(st.session_state.get("p_fare", 85.0 if st.session_state.p_pclass == 1 else (13.0 if st.session_state.p_pclass == 2 else 7.5)))
    
    # SibSp / Parch based on companions
    if "Solo" in st.session_state.p_companions:
        input_dict["SibSp"] = 0
        input_dict["Parch"] = 0
        input_dict["IsAlone"] = 1
    elif "Pair" in st.session_state.p_companions:
        input_dict["SibSp"] = 1
        input_dict["Parch"] = 0
        input_dict["IsAlone"] = 0
        if "TicketGroup_Pair (2)" in input_dict:
            input_dict["TicketGroup_Pair (2)"] = 1
    elif "Small" in st.session_state.p_companions:
        input_dict["SibSp"] = 1
        input_dict["Parch"] = 2
        input_dict["IsAlone"] = 0
        if "TicketGroup_Small Group (3-4)" in input_dict:
            input_dict["TicketGroup_Small Group (3-4)"] = 1
    else: # Large
        input_dict["SibSp"] = 3
        input_dict["Parch"] = 3
        input_dict["IsAlone"] = 0
        if "TicketGroup_Large Group (5+)" in input_dict:
            input_dict["TicketGroup_Large Group (5+)"] = 1

    # One-hot encoded features
    if st.session_state.p_sex == "male" and "Sex_male" in input_dict:
        input_dict["Sex_male"] = 1

    if st.session_state.p_pclass == 2 and "Pclass_2" in input_dict:
        input_dict["Pclass_2"] = 1
    elif st.session_state.p_pclass == 3 and "Pclass_3" in input_dict:
        input_dict["Pclass_3"] = 1

    # Title
    t_col = f"Title_{st.session_state.p_title}"
    if t_col in input_dict:
        input_dict[t_col] = 1

    # Embarked
    if "Cherbourg" in st.session_state.p_embarked and "Embarked_Q" in input_dict:
        pass # Reference is C or S
    elif "Queenstown" in st.session_state.p_embarked and "Embarked_Q" in input_dict:
        input_dict["Embarked_Q"] = 1
    elif "Southampton" in st.session_state.p_embarked and "Embarked_S" in input_dict:
        input_dict["Embarked_S"] = 1

    # Create DataFrame for model
    input_df = pd.DataFrame([input_dict])[feature_cols]

    # Predict base probability
    base_prob_lr = lr_model.predict_proba(input_df)[0][1]
    base_prob_gb = gb_model.predict_proba(input_df)[0][1]
    
    # Historical Officer Adjustment (Port Lightoller vs Starboard Murdoch)
    # Historic data: Starboard male survival ~37%, Port male survival ~18%
    is_starboard = "Starboard" in st.session_state.p_side
    is_male = st.session_state.p_sex == "male"
    is_adult = st.session_state.p_age >= 14

    officer_modifier = 0.0
    if is_male and is_adult:
        if is_starboard:
            officer_modifier = +0.12 # Murdoch allowed men
        else:
            officer_modifier = -0.14 # Lightoller strictly refused men
    elif not is_male:
        if not is_starboard:
            officer_modifier = +0.05 # Women prioritized on Port

    # Final combined probability
    final_prob = float(np.clip(0.6 * base_prob_lr + 0.4 * base_prob_gb + officer_modifier, 0.02, 0.98))
    survived = final_prob >= 0.50

    col_gauge, col_verdict = st.columns([1, 1], gap="large")

    with col_gauge:
        # Gauge chart
        gauge_color = "#52b788" if survived else "#e63946"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_prob * 100,
            number={"suffix": "%", "font": {"size": 48, "color": "#ffd166", "family": "Cinzel"}},
            title={"text": "ΠΙΘΑΝΟΤΗΤΑ ΕΠΙΒΙΩΣΗΣ", "font": {"size": 20, "color": "#ffffff", "family": "Cinzel"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ffffff"},
                "bar": {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "#121d2b",
                "borderwidth": 1,
                "bordercolor": "#445566",
                "steps": [
                    {"range": [0, 30], "color": "rgba(230, 57, 70, 0.25)"},
                    {"range": [30, 60], "color": "rgba(244, 162, 97, 0.25)"},
                    {"range": [60, 100], "color": "rgba(82, 183, 136, 0.25)"}
                ],
                "threshold": {
                    "line": {"color": "#ffd166", "width": 4},
                    "thickness": 0.8,
                    "value": 50
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff"},
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_verdict:
        if survived:
            render_html(f"""
            <div style="background: rgba(46, 117, 89, 0.25); border: 2px solid #52b788; border-radius: 12px; padding: 20px; margin-top: 15px;">
                <h2 style="color: #52b788; margin-top: 0; font-size: 1.5rem;">🎉 ΕΠΙΒΙΩΣΑΤΕ!</h2>
                <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                    Με πιθανότητα επιβίωσης <b>{final_prob:.1%}</b>, καταφέρατε να επιβιβαστείτε εγκαίρως σε μία από τις σωσίβιες λέμβους.
                    Περισυλλεγήκατε από το πλοίο <i>R.M.S. Carpathia</i> τα ξημερώματα της 15ης Απριλίου 1912.
                </p>
            </div>
            """)
        else:
            render_html(f"""
            <div style="background: rgba(186, 24, 27, 0.25); border: 2px solid #e63946; border-radius: 12px; padding: 20px; margin-top: 15px;">
                <h2 style="color: #e63946; margin-top: 0; font-size: 1.5rem;">❄️ ΧΑΘΗΚΑΤΕ ΣΤΟΝ ΑΤΛΑΝΤΙΚΟ</h2>
                <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                    Με πιθανότητα επιβίωσης μόλις <b>{final_prob:.1%}</b>, οι συνθήκες ήταν συντριπτικά εναντίον σας.
                    Το όνομά σας καταγράφηκε ανάμεσα στις 1.500 τραγικές απώλειες του ναυαγίου.
                </p>
            </div>
            """)

        factor_gender = 'Ευνοϊκό (Γυναίκα / Παιδί)' if not is_male or not is_adult else 'Δυσμενές (Ενήλικος άνδρας)'
        factor_class = 'Υψηλή προτεραιότητα' if st.session_state.p_pclass==1 else ('Μέτρια' if st.session_state.p_pclass==2 else 'Χαμηλή / Αποκλεισμός')
        factor_side = '+12% (Murdoch)' if is_starboard and is_male else ('-14% αποκλεισμός (Lightoller)' if not is_starboard and is_male else 'Ουδέτερο')

        render_html(f"""
        <div class="info-card" style="margin-top: 15px;">
            <b>🔑 Κύριοι Παράγοντες Επιρροής:</b><br>
            • <b>Φύλο & Ηλικία:</b> {factor_gender}<br>
            • <b>Θέση:</b> {st.session_state.p_pclass}η Θέση ({factor_class})<br>
            • <b>Πλευρά & Αξιωματικός:</b> {st.session_state.p_side} ({factor_side})
        </div>
        """)

        # Historical Dataset Comparison (αν πρόκειται για υπαρκτό επιβάτη του Kaggle dataset)
        if st.session_state.get("p_dataset_id") is not None:
            actual_surv = st.session_state.p_actual_survived
            match_model = (survived and actual_surv == 1) or (not survived and actual_surv == 0)
            status_color = "#52b788" if match_model else "#ffd166"
            verdict_text = "ΣΥΜΦΩΝΙΑ ΜΟΝΤΕΛΟΥ & ΙΣΤΟΡΙΑΣ" if match_model else "ΙΣΤΟΡΙΚΗ ΑΠΟΚΛΙΣΗ / ΕΞΑΙΡΕΣΗ"
            actual_str = f"ΕΠΕΖΗΣΕ ({st.session_state.p_boat})" if actual_surv == 1 else "ΧΑΘΗΚΕ ΣΤΟ ΝΑΥΑΓΙΟ"
            
            render_html(f"""
            <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid {status_color}; border-radius: 10px; padding: 14px 16px; margin-top: 15px;">
                <div style="font-size: 0.82rem; color: {status_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">
                    🏛️ Σύγκριση με το Kaggle Dataset (#{st.session_state.p_dataset_id}) • {verdict_text}
                </div>
                <div style="font-size: 0.92rem; color: #ffffff; line-height: 1.5;">
                    • <b>Πρόβλεψη Αλγορίθμου:</b> {final_prob:.1%} πιθανότητα ({'Επιβίωση' if survived else 'Απώλεια'})<br>
                    • <b>Πραγματική Έκβαση (1912):</b> <b style="color: {'#52b788' if actual_surv == 1 else '#e63946'};">{actual_str}</b>
                </div>
            </div>
            """)

    # Explainability Waterfall / Feature Impact Chart
    st.markdown("#### 🔬 Επεξήγηση Μοντέλου Μηχανικής Μάθησης (Feature Importance Breakdown)")
    st.caption("Πόσο συνέβαλε το κάθε σας χαρακτηριστικό στην τελική πιθανότητα επιβίωσης (βάσει συντελεστών Logistic Regression).")

    impacts = []
    for col, val in input_dict.items():
        if val != 0 and col in coefficients:
            weight = coefficients[col] * val
            readable_col = col.replace("Sex_male", "Φύλο (Άνδρας)").replace("Pclass_2", "2η Θέση").replace("Pclass_3", "3η Θέση").replace("IsAlone", "Μόνος/η χωρίς οικογένεια").replace("TicketGroup_Pair (2)", "Ζευγάρι (Pair)").replace("TicketGroup_Large Group (5+)", "Μεγάλη Ομάδα (5+)").replace("Embarked_S", "Southampton").replace("Embarked_Q", "Queenstown")
            impacts.append({"Feature": readable_col, "Impact": weight})

    if officer_modifier != 0.0:
        impacts.append({
            "Feature": f"Πλευρά Πλοίου ({'Starboard / Murdoch' if is_starboard else 'Port / Lightoller'})",
            "Impact": officer_modifier * 4.0
        })

    df_impacts = pd.DataFrame(impacts).sort_values("Impact")
    df_impacts["Color"] = df_impacts["Impact"].apply(lambda x: "#52b788" if x > 0 else "#e63946")

    fig_imp = px.bar(
        df_impacts,
        x="Impact",
        y="Feature",
        orientation="h",
        color="Impact",
        color_continuous_scale=[[0, "#e63946"], [0.5, "#445566"], [1, "#52b788"]],
        title="Συνεισφορά Χαρακτηριστικών στην Πιθανότητα Επιβίωσης (Log-Odds Impact)"
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18, 30, 49, 0.4)",
        font=dict(color="#ffffff"),
        coloraxis_showscale=False,
        height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Navigation buttons
    col_prev, _, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("⬅️ Πίσω: Προσομοίωση", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col_next:
        if st.button("Επόμενο: Key Notes & Στατιστικά ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 5: HISTORICAL KEY NOTES & RESEARCH DOSSIER
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 5:
    st.markdown("### 📊 Βήμα 5: Key Notes & Ερευνητικά Στατιστικά Ευρήματα")
    st.markdown("""
    Ανακαλύψτε τα βαθύτερα στατιστικά μοτίβα από την ανάλυση των πραγματικών δεδομένων του Τιτανικού.
    Αυτά τα ευρήματα φωτίζουν άγνωστες πτυχές της τραγωδίας.
    """)

    # Εμφάνιση του ενεργού επιβάτη & των ιστορικών insights του dataset
    render_passenger_banner(5)

    p_display = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "Επιβάτης Τιτανικού"
    st.info(f"💡 **Εστίαση στον Επιβάτη:** Ο/Η **{p_display}** ταξίδευε στην **{st.session_state.p_pclass}η Θέση** ({st.session_state.p_sex}, {st.session_state.p_age} ετών). Δείτε στις παρακάτω καρτέλες πώς η ταξική του θέση και το δημογραφικό του προφίλ επηρέασαν στατιστικά τη μοίρα του.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🛶 Το Παράδοξο των Λέμβων (472 Κενές Θέσεις)",
        "🎟️ Η Δυναμική των Εισιτηρίων (Ticket Groups)",
        "⚖️ Το Ταξικό Χάσμα (Class Disparity)",
        "👶 Γυναίκες & Παιδιά Πρώτα",
        "🔬 Η Επιστήμη του Ναυαγίου (Bulkheads, Rivets & Ασύρματος)",
        "📚 Βιβλιογραφία & Citations (Αναφορές)"
    ])

    with tab1:
        st.markdown("#### 🛶 Η Ανωμαλία των Σωσιβίων Λέμβων: 472 Κενές Θέσεις")
        st.markdown("""
        Ενώ ο Τιτανικός διέθετε λέμβους για **1.178 άτομα**, διασώθηκαν μόλις **706 επιβάτες**. 
        Αυτό σημαίνει ότι **472 θέσεις στις λέμβους έμειναν άδειες** και χάθηκαν στον ωκεανό!
        """)

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Συνολική Χωρητικότητα", "1,178 θέσεις")
        col_b2.metric("Πραγματικοί Επιβάτες", "706 άτομα", delta="-472 κενές θέσεις", delta_color="inverse")
        col_b3.metric("Μέση Πληρότητα Λέμβων", "60.4%")

        # Lifeboats bar chart
        df_boats_plot = df_boats.copy()
        df_boats_plot["Fill_Rate"] = (df_boats_plot["loaded"] / df_boats_plot["capacity"] * 100).round(1)

        fig_lb = px.bar(
            df_boats_plot,
            x="boat",
            y="Fill_Rate",
            color="side",
            color_discrete_map={"Port": "#48cae4", "Starboard": "#52b788"},
            labels={"boat": "Αριθμός Λέμβου", "Fill_Rate": "Ποσοστό Πληρότητας (%)", "side": "Πλευρά Πλοίου"},
            title="Πληρότητα ανά Σωσίβια Λέμβο (Port vs Starboard)"
        )
        fig_lb.add_hline(y=100, line_dash="dash", line_color="#ffd166", annotation_text="Μέγιστη Χωρητικότητα (100%)")
        fig_lb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 30, 49, 0.4)",
            font=dict(color="#ffffff"),
            height=360
        )
        st.plotly_chart(fig_lb, use_container_width=True)

        st.caption("📌 **Γιατί έμειναν άδειες;** 1) Οι πρώτοι αξιωματικοί φοβούνταν ότι οι λέμβοι θα έσπαγαν αν γέμιζαν πλήρως πριν κατέβουν στο νερό. 2) Δεν είχαν γίνει ποτέ ασκήσεις εκκένωσης για το πλήρωμα. 3) Πολλοί επιβάτες 1ης θέσης αρνούνταν να μοιραστούν τις λέμβους με επιβάτες άλλων τάξεων.")

    with tab2:
        st.markdown("#### 🎟️ Ticket Groups: Πώς το Μέγεθος της Παρέας Καθόρισε τη Ζωή")
        st.markdown("""
        Από την πρωτογενή ανάλυση των κοινών εισιτηρίων (Duplicate Tickets), προκύπτει ένα εντυπωσιακό μοτίβο: 
        **Τα ζευγάρια είχαν το υψηλότερο ποσοστό επιβίωσης (52%+)**, ενώ οι **μεγάλες ομάδες (5+ άτομα) κατέρρευσαν σε ποσοστά κάτω από 16%**!
        """)

        # Ticket Group Data from Notebook
        ticket_stats = pd.DataFrame({
            "Μέγεθος Ομάδας": ["Solo (Μόνος/η - 1)", "Pair (Ζευγάρι - 2)", "Small Group (3-4 άτομα)", "Large Group (5+ άτομα)"],
            "Ποσοστό Επιβίωσης (%)": [30.0, 53.2, 43.8, 15.6],
            "Πλήθος Επιβατών": [545, 188, 89, 69]
        })

        fig_tg = px.bar(
            ticket_stats,
            x="Μέγεθος Ομάδας",
            y="Ποσοστό Επιβίωσης (%)",
            color="Ποσοστό Επιβίωσης (%)",
            color_continuous_scale="Viridis",
            text="Ποσοστό Επιβίωσης (%)",
            title="Ποσοστό Επιβίωσης ανά Μέγεθος Ομάδας Εισιτηρίου"
        )
        fig_tg.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_tg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 30, 49, 0.4)",
            font=dict(color="#ffffff"),
            coloraxis_showscale=False,
            height=340
        )
        st.plotly_chart(fig_tg, use_container_width=True)

        st.caption("💡 **Κοινωνική Ερμηνεία**: Στα ζευγάρια, ο άνδρας συνήθως εξασφάλιζε θέση για τη σύζυγό του στη λέμβο. Αντίθετα, στις πολύτεκνες οικογένειες (κυρίως 3ης θέσης), οι γονείς αρνήθηκαν να χωριστούν από τα παιδιά τους, με αποτέλεσμα να χαθούν ολόκληρες οικογένειες μαζί (όπως η οικογένεια Goodwin με 8 μέλη και η οικογένεια Sage με 11 μέλη).")

    with tab3:
        st.markdown("#### ⚖️ Το Κοινωνικό και Ταξικό Χάσμα (1η vs 2η vs 3η Θέση)")
        st.markdown("""
        Η ταξική ανισότητα στον Τιτανικό ήταν καταλυτική. Οι επιβάτες 1ης θέσης είχαν **υπερδιπλάσιες πιθανότητες επιβίωσης** σε σύγκριση με την 3η θέση.
        """)

        class_summary = pd.DataFrame({
            "Θέση (Class)": ["1η Θέση (First)", "2η Θέση (Second)", "3η Θέση (Third)"],
            "Επιβίωση (%)": [62.9, 47.3, 24.2],
            "Επιβίωση Γυναικών (%)": [96.8, 92.1, 50.0],
            "Επιβίωση Ανδρών (%)": [36.9, 15.7, 13.5]
        })

        st.dataframe(class_summary, use_container_width=True, hide_index=True)

        fig_cls = px.bar(
            class_summary,
            x="Θέση (Class)",
            y=["Επιβίωση Γυναικών (%)", "Επιβίωση Ανδρών (%)"],
            barmode="group",
            title="Ποσοστά Επιβίωσης ανά Φύλο και Κοινωνική Θέση",
            color_discrete_sequence=["#ffd166", "#48cae4"]
        )
        fig_cls.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 30, 49, 0.4)",
            font=dict(color="#ffffff"),
            height=340
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    with tab4:
        st.markdown("#### 👶 Το Πρωτόκολλο 'Γυναίκες και Παιδιά Πρώτα' & Η Τραγωδία των Τάξεων")
        st.markdown("""
        Το ιστορικό ναυτικό πρωτόκολλο τηρήθηκε πιστά, αλλά η πιθανότητα σωτηρίας ενός παιδιού κρίθηκε συντριπτικά από το εισιτήριό του:
        - **2η Θέση**: **100% επιβίωση των παιδιών** (18 στα 18 παιδιά στο Kaggle dataset, 24 στα 24 ιστορικά).
        - **1η Θέση**: **3 στα 4 παιδιά επιβίωσαν** (75% στο dataset, 5 στα 6 ιστορικά). Το μοναδικό παιδί 1ης/2ης θέσης που χάθηκε ήταν η 2χρονη **Helen Lorraine Allison**, καθώς οι γονείς της αρνήθηκαν να επιβιβαστούν σε λέμβο ψάχνοντας απεγνωσμένα το βρέφος τους (το οποίο είχε ήδη διασωθεί από τη νταντά του).
        - **3η Θέση**: **Μόλις το 42.9% των παιδιών επιβίωσε** (21 στα 49 στο dataset), καθώς πάνω από 50 παιδιά 3ης θέσης παγιδεύτηκαν στα κατώτερα καταστρώματα.
        """)

        child_stats = pd.DataFrame({
            "Κατηγορία Παιδιών (< 14 ετών στο Dataset)": ["Παιδιά 1ης Θέσης", "Παιδιά 2ης Θέσης", "Παιδιά 3ης Θέσης"],
            "Ποσοστό Επιβίωσης (%)": [75.0, 100.0, 42.9],
            "Επέζησαν": [3, 18, 21],
            "Χάθηκαν": [1, 0, 28]
        })
        st.dataframe(child_stats, use_container_width=True, hide_index=True)

        st.caption("📌 **Πηγή Δεδομένων**: Kaggle Titanic Dataset (`yasserh/titanic-dataset` / Kaggle Titanic ML).")

    with tab5:
        st.markdown("#### 🔬 Η Επιστήμη του Ναυαγίου: Γιατί Βυθίστηκε το 'Αβύθιστο' Πλοίο;")
        st.markdown("""
        Με βάση τις επιστημονικές και ναυπηγικές αναλύσεις (αρχείο *Titanic Science*):
        """)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">1. Το Φαινόμενο της Παγοθήκης (Bulkheads)</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    Ο Τιτανικός διέθετε <b>16 στεγανά διαμερίσματα</b> και είχε σχεδιαστεί να παραμένει στην επιφάνεια ακόμα και με <b>4 διαμερίσματα πλημμυρισμένα</b>. 
                    Το παγόβουνο έπληξε <b>6 διαμερίσματα</b> στην πλώρη.<br><br>
                    <b>Το μοιραίο σχεδιαστικό λάθος:</b> Τα διαφράγματα δεν ήταν σφραγισμένα στην οροφή, αλλά έφταναν μόνο μέχρι το Κατάστρωμα E (ή D στην πλώρη). 
                    Καθώς το βάρος του νερού έγειρε την πλώρη προς τα κάτω, το νερό υπερχείλισε πάνω από το πρώτο διάφραγμα στο επόμενο, 
                    <b>ακριβώς όπως γεμίζει μια παγοθήκη κάτω από τη βρύση</b>!
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">2. Η Αστοχία των Ήλων (Rivet Failure)</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    Το κύτος συγκρατούνταν από <b>3.000.000 πριτσίνια (ήλους)</b>, συνολικού βάρους 1.200 τόνων. 
                    Μεταλλουργικές εξετάσεις στα συντρίμμια απέδειξαν ότι τα πριτσίνια της πλώρης περιείχαν υψηλή συγκέντρωση σκωρίας (slag). 
                    Στη θερμοκρασία του παγωμένου νερού (-2.8°C / 27°F), το μέταλλο έγινε <b>ψαθυρό (brittle)</b> και τα κεφάλια των πριτσινιών αποκόπηκαν 
                    υπό την πίεση του παγόβουνου, ανοίγοντας τις ραφές των χαλύβδινων πλακών.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">3. Η Τραγωδία των Ασυρμάτων & Το 'SS Californian'</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    Το πλησιέστερο πλοίο στον Τιτανικό τη νύχτα του ναυαγίου ήταν το <b>SS Californian</b>, σε απόσταση μόλις <b>10 έως 19 ναυτικών μιλίων</b> 
                    (οι επιβάτες του Τιτανικού έβλεπαν τα φώτα του στον ορίζοντα!).<br><br>
                    <b>Γιατί δεν βοήθησε;</b> Ο μοναδικός ασυρματιστής του Californian, Cyril Evans, έκλεισε τον ασύρματο και πήγε για ύπνο στις 23:30 
                    (μόλις 10 λεπτά πριν τη σύγκρουση). Οι ναύτες είδαν τις λευκές φωτοβολίδες του Τιτανικού αλλά δεν κατάλαβαν ότι ήταν σήματα κινδύνου.<br><br>
                    Αντίθετα, το <b>RMS Carpathia</b> βρισκόταν 58 μίλια μακριά, έπλευσε με τη μέγιστη ταχύτητα των 17.5 κόμβων μέσα στα παγόβουνα 
                    και έφτασε στις 04:00 για να περισυλλέξει τους 705 επιζώντες.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">4. Υποθερμία σε Παγωμένα Νερά (-2.8°C)</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    Η θερμοκρασία του νερού ήταν <b>-2.8°C (27°F)</b>. 
                    Σε τέτοιες συνθήκες, η θερμοκρασία του σώματος πέφτει ραγδαία κάτω από τους 35°C (υποθερμία). 
                    Ο θάνατος επερχόταν μέσα σε <b>10 έως 15 λεπτά</b>, καθιστώντας την επιβίωση στο νερό χωρίς λέμβο αδύνατη. 
                    Από τους εκατοντάδες που έπεσαν στη θάλασσα, λιγότεροι από 6 διασώθηκαν ζωντανοί.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab6:
        st.markdown("#### 📚 Επιστημονική & Ιστορική Βιβλιογραφία (References & Citations)")
        st.markdown("""
        Όλα τα δεδομένα, οι μαθηματικές συναρτήσεις, οι ιστορικές αναπαραστάσεις των λέμβων και τα φυσικά χαρακτηριστικά του πλοίου 
        βασίζονται σε διεθνώς αναγνωρισμένες επιστημονικές πηγές, κρατικές ανακρίσεις και ακαδημαϊκή βιβλιογραφία (μορφότυπο APA):
        """)

        st.markdown("""
        ##### 📊 1. Σύνολα Δεδομένων & Πηγές Μηχανικής Μάθησης (Datasets)
        - **Kaggle Titanic Dataset**: Yasser, H. (2021). *Titanic Dataset*. Kaggle.  
          URL: [https://www.kaggle.com/datasets/yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
        - **Kaggle Competition Data**: Kaggle Inc. (2012). *Titanic: Machine Learning from Disaster*.  
          URL: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
        - **Encyclopedia Titanica Manifest & Archives**: Hind, P. et al. (1996–2026). *Encyclopedia Titanica: Titanic Passenger Manifests, Deck Plans & Lifeboat Allocations*.  
          URL: [https://www.encyclopedia-titanica.org](https://www.encyclopedia-titanica.org)

        ##### 🏛️ 2. Επίσημες Κρατικές Ανακρίσεις (Official Governmental Inquiries, 1912)
        - **Βρετανική Ανάκριση (British Mersey Inquiry)**:  
          Mersey, J. C. B., Baron. (1912). *Report on the Loss of the "Titanic" (S.S.): Formal Investigation into the Loss of the British S.S. Titanic*. Parliamentary Papers, Cd. 6352. London: His Majesty's Stationery Office.
        - **Αμερικανική Ανάκριση (United States Senate Inquiry)**:  
          Smith, W. A. et al. (1912). *"Titanic" Disaster: Hearings before a Subcommittee of the Committee on Commerce, United States Senate, Sixty-Second Congress, Second Session* (Senate Document No. 726 / Report 806). Washington: Government Printing Office.

        ##### 📖 3. Αυτόπτες Μάρτυρες & Απομνημονεύματα Επιζώντων (Survivor Eyewitness Accounts)
        - **Lawrence Beesley (Επιβάτης 2ης Θέσης)**:  
          Beesley, L. (1912). *The Loss of the SS Titanic: Its Story and Its Lessons*. Boston & New York: Houghton Mifflin Company.
        - **John Borland "Jack" Thayer Jr. (Επιβάτης 1ης Θέσης)**:  
          Thayer, J. B., Jr. (1940). *The Sinking of the S.S. Titanic: An Event That Shook the World*. Philadelphia: The Academy of Natural Sciences.
        - **Charles Herbert Lightoller (2ος Αξιωματικός)**:  
          Lightoller, C. H. (1935). *Titanic and Other Ships*. London: Ivor Nicholson and Watson.

        ##### 🔬 4. Μεταλλουργική & Ναυπηγική Εγκληματολογία (Forensic & Metallurgical Research)
        - **Αστοχία Ήλων & Σκωρία (Rivet Metallurgy)**:  
          Foecke, T., & McCarty, J. (2008). *What Really Sank the Titanic: New Forensic Discoveries*. New York: Citadel Press.
        - **Δομική Κατάρρευση & Υδροδυναμική Βύθισης**:  
          Garzke, W. H., Brown, D. K., & Woodward, J. B. (1997). *The Structural Failure of the Titanic: Investigation and Analysis*. SNAME Transactions (Society of Naval Architects and Marine Engineers), 105, 237–264.

        ##### 💻 5. Εργαλεία Λογισμικού & Αλγόριθμοι Μηχανικής Μάθησης (Software Libraries)
        - **Scikit-learn**: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
        - **Streamlit**: Snowflake Inc. (2026). *Streamlit: A faster way to build and share data apps*. Open source framework. URL: [https://streamlit.io](https://streamlit.io)
        - **Plotly**: Plotly Technologies Inc. (2026). *Collaborative Data Science and 3D Visualizations*. URL: [https://plotly.com](https://plotly.com)
        """)

    # Completion banner & Navigation buttons
    st.info("🏁 **Ολοκληρώσατε όλη τη διαδρομή (Βήμα 5 από 5)!** Μπορείτε να ανατρέξετε στα προηγούμενα βήματα ή να πατήσετε **'Συνέχεια: Νέος Επιβάτης'** για να δοκιμάσετε ένα νέο σενάριο.")

    col_prev, _, col_restart = st.columns([1, 2, 2])
    with col_prev:
        if st.button("⬅️ Πίσω: Πιθανότητα Επιβίωσης", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
    with col_restart:
        if st.button("Συνέχεια: Νέος Επιβάτης (Βήμα 1) 🔄", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER & CREDITS & DATASET DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📖 Λεξικό Μεταβλητών & Τεκμηρίωση Dataset (Kaggle: yasserh/titanic-dataset)"):
    st.markdown("""
    Το μοντέλο και τα στατιστικά βασίζονται στο διάσημο **Kaggle Titanic Dataset** (`yasserh/titanic-dataset` / Titanic ML from Disaster) με 891 επιβάτες:

    | Μεταβλητή | Περιγραφή | Τιμές / Επεξήγηση |
    | :--- | :--- | :--- |
    | **`PassengerId`** | Μοναδικό αναγνωριστικό επιβάτη | 1 έως 891 |
    | **`Survived`** | Κατάσταση Επιβίωσης (Target) | `0` = Απώλεια, `1` = Επιβίωση |
    | **`Pclass`** | Κοινωνικοοικονομική Τάξη (Class) | `1` = 1η Θέση (Upper), `2` = 2η (Middle), `3` = 3η (Lower) |
    | **`Name`** | Πλήρες όνομα και κοινωνικός τίτλος | Περιλαμβάνει τίτλους όπως Mr, Mrs, Miss, Master, Col κ.ά. |
    | **`Sex`** | Βιολογικό Φύλο | `male`, `female` |
    | **`Age`** | Ηλικία σε έτη | Κλασματική αν < 1 έτους (π.χ. 0.92) |
    | **`SibSp`** | Αδέλφια ή σύζυγοι στο πλοίο | `Siblings` (αδέλφια/ετεροθαλή) & `Spouses` (σύζυγοι) |
    | **`Parch`** | Γονείς ή τέκνα στο πλοίο | `Parents` (μητέρα/πατέρας) & `Children` (γιος/κόρη) |
    | **`Ticket`** | Κωδικός εισιτηρίου | Αλφαριθμητικό (κοινό εισιτήριο για ζευγάρια/οικογένειες) |
    | **`Fare`** | Τιμή ναύλου εισιτηρίου | Ποσό σε λίρες Αγγλίας (£) του 1912 |
    | **`Cabin`** | Αριθμός καμπίνας και κατάστρωμα | Γράμμα καταστρώματος (A-G) και αριθμός δωματίου |
    | **`Embarked`** | Λιμάνι επιβίβασης | `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |
    """)

render_html("""
<div style="text-align: center; margin-top: 35px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.85rem; color: #8fa3b8; line-height: 1.6;">
    ⚓ <b>Titanic Interactive 3D Survival Simulator</b><br>
    Επιστημονικές & Ιστορικές Πηγές: <a href="https://www.kaggle.com/datasets/yasserh/titanic-dataset" target="_blank" style="color: #48cae4; text-decoration: none;">Kaggle Titanic Dataset</a> • <a href="https://www.encyclopedia-titanica.org" target="_blank" style="color: #48cae4; text-decoration: none;">Encyclopedia Titanica</a> • Mersey Inquiry (1912) • US Senate (1912)<br>
    Machine Learning Architecture: Scikit-Learn (Logistic Regression & Gradient Boosting) • 3D Visualizations: Plotly
</div>
""")
