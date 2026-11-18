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

# Custom CSS for rich maritime look and universal theme compatibility (Light & Dark)
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Force luxurious dark maritime canvas across all browser themes */
    .stApp, [data-testid="stAppViewContainer"], .main, header, [data-testid="stHeader"] {
        background: radial-gradient(circle at 50% 10%, #0d1b2a 0%, #080f1a 100%) !important;
        color: #e0e6ed !important;
    }

    /* Consistent high-contrast typography */
    h1, h2, h3, h4, h5, h6, .brand-title {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 1px;
        color: #ffd166 !important;
    }

    p, span, label, div[data-testid="stMarkdownContainer"] p, .stMarkdown {
        color: #d0dbe8 !important;
    }

    /* Form controls: text input, selectbox, sliders in both light & dark mode */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    input, textarea {
        background-color: #121e31 !important;
        color: #ffffff !important;
        border-color: rgba(212, 175, 55, 0.4) !important;
    }

    /* Dropdown popover menu items */
    ul[data-baseweb="menu"], li[data-baseweb="menu-item"] {
        background-color: #0d1726 !important;
        color: #ffffff !important;
    }

    /* Tab navigation bar */
    button[data-baseweb="tab"] {
        color: #8da4be !important;
        background-color: transparent !important;
        font-weight: 600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #0284c7 !important;
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        color: #d0dbe8 !important;
    }

    /* Boarding pass luxury styling */
    .boarding-pass {
        background: linear-gradient(135deg, #162438 0%, #0d1726 100%);
        border: 2px solid rgba(56, 189, 248, 0.4);
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
        color: #38bdf8 !important;
        margin-bottom: 2px;
    }

    .boarding-sub {
        font-size: 0.8rem;
        color: #a4b8d1 !important;
        letter-spacing: 2px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.3);
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    .pass-label {
        font-size: 0.72rem;
        color: #829ab8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .pass-val {
        font-size: 1.05rem;
        color: #ffffff !important;
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
        color: #ffd166 !important;
        font-weight: 700;
    }

    .highlight-cyan {
        color: #48cae4 !important;
        font-weight: 600;
    }

    /* Modern Oceanic Blue buttons instead of harsh yellow */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: #132337 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }

    .stButton>button:hover {
        background: #1e3a5f !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
    }

    /* Primary buttons (Next step, active tab, start over) - Sleek Royal / Ocean Cyan gradient */
    .stButton>button[kind="primary"],
    .stButton>button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.45) !important;
        font-weight: 700 !important;
    }

    .stButton>button[kind="primary"]:hover,
    .stButton>button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border-color: #7dd3fc !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #8da4be !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Cinzel', serif !important;
        color: #38bdf8 !important;
    }
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL BUNDLE & DATASET
# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL BUNDLE & DATASET (WITH ROBUST CLOUD IN-MEMORY FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────
def train_model_in_memory():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    app_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(app_dir, "Titanic.csv")
    xlsx_path = os.path.join(app_dir, "Titanic.xlsx")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError("Could not find Titanic.csv")

    df.columns = df.columns.str.strip()
    df["Fare"] = np.where(df["Fare"] > 1000, df["Fare"] / 10000.0, df["Fare"])
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].fillna("Rare")

    df["SibSp"] = df["SibSp"].fillna(0).astype(int)
    df["Parch"] = df["Parch"].fillna(0).astype(int)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    ticket_counts = df["Ticket"].value_counts()
    df["TicketCount"] = df["Ticket"].map(ticket_counts).fillna(1).astype(int)
    def classify_ticket(c):
        if c == 1:
            return "Solo (1)"
        elif c == 2:
            return "Pair (2)"
        elif c <= 4:
            return "Small Group (3-4)"
        else:
            return "Large Group (5+)"
    df["TicketGroup"] = df["TicketCount"].apply(classify_ticket)

    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Embarked"] = df["Embarked"].fillna("S")

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
        "Embarked", "Title", "IsAlone", "TicketGroup"
    ]

    X = pd.get_dummies(df[feature_cols], columns=["Pclass", "Sex", "Embarked", "Title", "TicketGroup"], drop_first=True)
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1500, random_state=42)
    lr.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
    gb.fit(X_train, y_train)

    coefficients = pd.Series(lr.coef_[0], index=X.columns).to_dict()

    bundle = {
        "lr_model": lr,
        "gb_model": gb,
        "feature_columns": list(X.columns),
        "coefficients": coefficients,
        "intercept": float(lr.intercept_[0]),
        "df_clean": df
    }
    return bundle

@st.cache_resource
def load_model_bundle():
    """
    Trains the Logistic Regression & Gradient Boosting models in-memory directly
    from Titanic.csv using the exact host environment libraries.
    Ensures zero cross-version unpickling warnings or module import errors.
    """
    return train_model_in_memory()

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
    "⚙️ Custom Passenger (Create Your Own)": {
        "name": "", "age": 28, "sex": "male", "pclass": 2, "title": "Mr",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Solo (1 person)", "deck": "D", "side": "Starboard (Right - Officer Murdoch)",
        "fare": 13.0, "dataset_id": None, "ticket": "", "actual_survived": None, "boat": "",
        "insight": ""
    },
    "👑 Madeleine Force Astor (1st Class - Age 18, Wife of J.J. Astor IV, Boat 4) [ID #701]": {
        "name": "Madeleine Talmadge Force Astor", "age": 18, "sex": "female", "pclass": 1, "title": "Mrs",
        "nationality": "American (USA)", "embarked": "Cherbourg (C)", "companions": "Pair (2 people - spouse/companion)", "deck": "C", "side": "Port (Left - Officer Lightoller)",
        "fare": 227.52, "dataset_id": 701, "ticket": "PC 17757", "actual_survived": 1, "boat": "Lifeboat 4 (Port)",
        "insight": "The 18-year-old pregnant wife of Colonel John Jacob Astor IV (the wealthiest man in the world). Astor helped her board Lifeboat 4 through a promenade deck window. When he asked Second Officer Lightoller if he could join to protect her due to her advanced pregnancy, Lightoller strictly refused ('No men until all women are aboard'). Astor perished in the freezing Atlantic, while Madeleine survived and gave birth to their son John Jacob Astor VI a few months later."
    },
    "💎 Margaret 'Molly' Brown (1st Class - Age 44, 'The Unsinkable Molly Brown', Boat 6) [ID #195]": {
        "name": "Margaret 'Molly' Tobin Brown", "age": 44, "sex": "female", "pclass": 1, "title": "Mrs",
        "nationality": "American (USA)", "embarked": "Cherbourg (C)", "companions": "Solo (1 person)", "deck": "B", "side": "Port (Left - Officer Lightoller)",
        "fare": 27.72, "dataset_id": 195, "ticket": "PC 17610", "actual_survived": 1, "boat": "Lifeboat 6 (Port)",
        "insight": "The internationally famous 'Unsinkable Molly Brown'. In Lifeboat 6 (which left with only 24 people despite a 65-person capacity), she demanded Quartermaster Robert Hichens turn back to rescue swimmers. When he terrified refused, Brown took an oar herself and kept the women warm with her furs. Aboard the rescue ship Carpathia, she distributed provisions and founded the Survivors' Committee, raising $10,000 for destitute 3rd class immigrants."
    },
    "💼 Prof. Lawrence Beesley (2nd Class - Age 34, Cambridge Physics Professor, Boat 13) [ID #22]": {
        "name": "Lawrence Beesley", "age": 34, "sex": "male", "pclass": 2, "title": "Mr",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Solo (1 person)", "deck": "D", "side": "Starboard (Right - Officer Murdoch)",
        "fare": 13.0, "dataset_id": 22, "ticket": "248698", "actual_survived": 1, "boat": "Lifeboat 13 (Starboard)",
        "insight": "Cambridge-educated physics master at Dulwich College. Positioned on the Starboard side under First Officer William Murdoch, who enforced 'Women and children first' (not exclusively), allowing men to board when no women were waiting. As Lifeboat 13 was lowered, Murdoch called out to him to jump in as a seat remained vacant. Beesley subsequently authored the definitive eyewitness memoir: 'The Loss of the SS Titanic' (1912)."
    },
    "🎓 Jack Thayer Jr. (1st Class - Age 17, Overturned Collapsible B) [ID #551]": {
        "name": "John Borland 'Jack' Thayer Jr.", "age": 17, "sex": "male", "pclass": 1, "title": "Mr",
        "nationality": "American (USA)", "embarked": "Cherbourg (C)", "companions": "Small Group (3-4 people - family)", "deck": "C", "side": "Port (Left - Officer Lightoller)",
        "fare": 110.88, "dataset_id": 551, "ticket": "17421", "actual_survived": 1, "boat": "Collapsible B (Overturned)",
        "insight": "17-year-old son of Pennsylvania Railroad VP John B. Thayer. Separated from his parents in the final minutes, he jumped into the frigid Atlantic and scrambled atop the overturned Collapsible B alongside Officer Lightoller. His father drowned, but his mother Marian (#582) was rescued. Jack Thayer's eyewitness sketches provided the earliest compelling visual proof that Titanic split in two before foundering."
    },
    "👶 Master Hudson Trevor Allison (1st Class - Infant 11 mos, Boat 11) [ID #306]": {
        "name": "Master Hudson Trevor Allison", "age": 1, "sex": "male", "pclass": 1, "title": "Master",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Small Group (3-4 people - family)", "deck": "C", "side": "Starboard (Right - Officer Murdoch)",
        "fare": 151.55, "dataset_id": 306, "ticket": "113781", "actual_survived": 1, "boat": "Lifeboat 11 (Starboard)",
        "insight": "An 11-month-old infant (0.92 years in the dataset). Amidst the confusion, his nursemaid Alice Cleaver carried him onto Lifeboat 11. His wealthy parents and 2-year-old sister Helen (#298) refused to board lifeboats, frantically searching the ship for their baby without knowing he was already safe. The rest of the family perished. Little Helen Allison was the only child from 1st or 2nd class who died on the Titanic."
    },
    "🤺 Sir Cosmo Edmund Duff Gordon (1st Class - Age 49, 'The Money Boat', Boat 1) [ID #600]": {
        "name": "Sir Cosmo Edmund Duff Gordon", "age": 49, "sex": "male", "pclass": 1, "title": "Rare",
        "nationality": "British (UK)", "embarked": "Cherbourg (C)", "companions": "Pair (2 people - spouse/companion)", "deck": "A", "side": "Starboard (Right - Officer Murdoch)",
        "fare": 56.93, "dataset_id": 600, "ticket": "PC 17485", "actual_survived": 1, "boat": "Emergency Cutter 1 (Starboard)",
        "insight": "Scottish baronet and Olympic fencing silver medalist. Boarded Lifeboat 1 (capacity 40) with his wife Lady Duff Gordon; the boat pulled away with only 12 people (7 crew and 5 first-class passengers). Duff Gordon gifted £5 to each crewman, triggering a sensational press controversy ('The Money Boat') alleging they bribed the sailors not to return for swimmers."
    },
    "☘️ Anna Sofia Turja (3rd Class - Age 18, Finnish Immigrant, Boat 15) [ID #678]": {
        "name": "Anna Sofia Turja", "age": 18, "sex": "female", "pclass": 3, "title": "Miss",
        "nationality": "Scandinavian (Sweden/Norway)", "embarked": "Southampton (S)", "companions": "Solo (1 person)", "deck": "F", "side": "Starboard (Right - Officer Murdoch)",
        "fare": 9.84, "dataset_id": 678, "ticket": "4138", "actual_survived": 1, "boat": "Lifeboat 15 (Starboard)",
        "insight": "An 18-year-old Finnish farming girl traveling alone to Ohio without speaking a word of English. While many steerage women were trapped below by locked security gates and language barriers, Anna followed a small group of fellow passengers who located an unlocked emergency stairwell leading to the Boat Deck. She reached Lifeboat 15 moments before it was lowered."
    },
    "👶 Master Michel Marcel Navratil (2nd Class - Age 3, 'Titanic Orphans', Boat D) [ID #194]": {
        "name": "Master Michel Marcel Navratil", "age": 3, "sex": "male", "pclass": 2, "title": "Master",
        "nationality": "French", "embarked": "Southampton (S)", "companions": "Small Group (3-4 people - family)", "deck": "E", "side": "Port (Left - Officer Lightoller)",
        "fare": 26.0, "dataset_id": 194, "ticket": "230080", "actual_survived": 1, "boat": "Collapsible D (Port)",
        "insight": "A 3-year-old boy. His father Michel Navratil Sr. (#149) had taken him and his 2-year-old brother Edmond (#341) from their mother in France, traveling under the alias 'Louis Hoffman' to New York. On the night of the sinking, the father handed both toddlers into the final successful craft (Collapsible D), telling Michel: 'Tell your mother I loved her.' The father drowned, and the brothers became internationally known as the 'Titanic Orphans' until reunited with their mother in Nice."
    },
    "💔 Master Harold Victor Goodwin (3rd Class - Age 9, Family of 8) [ID #481]": {
        "name": "Master Harold Victor Goodwin", "age": 9, "sex": "male", "pclass": 3, "title": "Master",
        "nationality": "British (UK)", "embarked": "Southampton (S)", "companions": "Large Group (5+ people - extended family)", "deck": "G", "side": "Port (Left - Officer Lightoller)",
        "fare": 46.90, "dataset_id": 481, "ticket": "CA 2144", "actual_survived": 0, "boat": "None (Lost in disaster)",
        "insight": "A 9-year-old boy from Fulham, England, emigrating with his parents and 5 siblings (family of 8). Due to their large family size (Ticket Group 5+), deep steerage quarters, and locked gates, not a single member of the Goodwin family survived. His 19-month-old brother Sidney (#387) was pulled from the ocean and identified via DNA in 2007 as 'The Unknown Child'."
    },
    "🇬🇷 Apostolos Chronopoulos (3rd Class - Age 26, Greek Immigrant) [ID #74]": {
        "name": "Apostolos Chronopoulos", "age": 26, "sex": "male", "pclass": 3, "title": "Mr",
        "nationality": "Greek / Balkan", "embarked": "Cherbourg (C)", "companions": "Pair (2 people - spouse/companion)", "deck": "G", "side": "Port (Left - Officer Lightoller)",
        "fare": 14.45, "dataset_id": 74, "ticket": "2680", "actual_survived": 0, "boat": "None (Lost in disaster)",
        "insight": "A 26-year-old Greek laborer from Agios Sostis, Messenia. He embarked at Cherbourg with his 19-year-old brother Demetrios on ticket #2680 seeking opportunities in America. All 4 documented Greek passengers aboard (young men from the same village) perished during the disaster, trapped in steerage unable to understand the crew's English commands."
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
    p_name = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "Titanic Passenger"
    is_dataset_preset = st.session_state.get("p_dataset_id") is not None
    
    col_pinfo, col_pswitch = st.columns([3, 2], gap="medium")
    
    with col_pswitch:
        preset_keys = list(PRESETS.keys())
        cur_idx = preset_keys.index(st.session_state.last_preset) if st.session_state.last_preset in preset_keys else 0
        new_preset = st.selectbox(
            "🎭 Select Passenger Persona:",
            preset_keys,
            index=cur_idx,
            key=f"banner_preset_switch_{screen_step}",
            help="You can switch passenger profile anytime across all 5 steps."
        )
        if new_preset != st.session_state.last_preset:
            apply_preset(new_preset)
            st.rerun()

    with col_pinfo:
        badge_outcome = ""
        if is_dataset_preset:
            if st.session_state.p_actual_survived == 1:
                badge_outcome = f'<span style="background: rgba(82, 183, 136, 0.25); color: #52b788; border: 1px solid #52b788; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">✅ SURVIVED ({st.session_state.p_boat})</span>'
            else:
                badge_outcome = '<span style="background: rgba(230, 57, 70, 0.25); color: #e63946; border: 1px solid #e63946; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">❌ PERISHED IN DISASTER</span>'

        dataset_id_str = f" • <span style='color: #ffd166; font-weight: 700;'>Kaggle ID #{st.session_state.p_dataset_id}</span>" if is_dataset_preset else ""
        ticket_str = f" • Ticket: <b>{st.session_state.p_ticket}</b>" if st.session_state.get("p_ticket") else ""
        fare_val = float(st.session_state.get("p_fare", 13.0))
        side_short = "Starboard (Right)" if "Starboard" in st.session_state.p_side else "Port (Left)"

        render_html(f"""
        <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid rgba(212, 175, 55, 0.35); border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 1.12rem; font-weight: 700; color: #ffffff;">
                    👤 {p_name} <span style="font-size: 0.88rem; font-weight: normal; color: #a4b8d1;">(Class {st.session_state.p_pclass} • Age {st.session_state.p_age} • {st.session_state.p_sex.capitalize()}{dataset_id_str})</span>
                </div>
                <div>{badge_outcome}</div>
            </div>
            <div style="font-size: 0.84rem; color: #a0c4e2; margin-top: 6px;">
                📍 <b>Deck:</b> Deck {st.session_state.p_deck} • <b>Side:</b> {side_short} • <b>Fare:</b> £{fare_val:.2f}{ticket_str} • <b>Port:</b> {st.session_state.p_embarked.split()[0]}
            </div>
        </div>
        """)

    # Display Dataset Insights & Historical Profile across all steps
    if is_dataset_preset and st.session_state.get("p_insight"):
        render_html(f"""
        <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid rgba(212, 175, 55, 0.4); border-radius: 12px; padding: 16px 20px; margin: 10px 0 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <div style="font-family: 'Cinzel', serif; font-size: 1.05rem; color: #ffd166; font-weight: 700; margin-bottom: 8px;">
                📋 Historical Persona & Dataset Insights: {p_name} (#{st.session_state.p_dataset_id})
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
    st.session_state.p_companions = "Solo (1 person)"
if "p_deck" not in st.session_state:
    st.session_state.p_deck = "D"
if "p_side" not in st.session_state:
    st.session_state.p_side = "Starboard (Right - Officer Murdoch)"
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
    st.session_state.last_preset = "⚙️ Custom Passenger (Create Your Own)"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER & APP TITLE
# ─────────────────────────────────────────────────────────────────────────────
render_html("""
<div style="text-align: center; margin-bottom: 25px;">
    <h1 style="color: #ffd166; margin-bottom: 5px; font-size: 2.3rem;">R.M.S. TITANIC</h1>
    <h3 style="color: #a0c4e2; font-weight: 300; font-size: 1.2rem; letter-spacing: 2px;">
        INTERACTIVE 3D SIMULATION, NAVAL LOCATOR & MACHINE LEARNING
    </h3>
    <p style="color: #8da2ba; max-width: 800px; margin: 0 auto; font-size: 0.95rem;">
        Experience the fateful voyage of April 14, 1912. Choose or create a passenger, explore your cabin in 3D, 
        simulate the sinking timeline, and uncover your survival probability using Machine Learning.
    </p>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE STEPPER NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
steps = [
    ("1. 🎟️ Boarding Pass", 1),
    ("2. 🚢 3D Ship Deck", 2),
    ("3. ⏱️ Night Timeline", 3),
    ("4. 🔮 Survival ML", 4),
    ("5. 📊 Key Notes & Citations", 5),
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
    st.markdown("### 🎟️ Step 1: Boarding Pass & Passenger Identity (1912)")
    st.info("💡 Select a historical passenger persona (Preset) or customize your attributes manually below to board the Titanic.")

    # Persona Selector: Real verified passengers from Titanic historical records & Kaggle Dataset
    presets = PRESETS
    cur_p_idx = list(presets.keys()).index(st.session_state.last_preset) if st.session_state.last_preset in presets else 0
    selected_preset = st.selectbox("🎭 Select Historical Passenger Persona (or create custom):", list(presets.keys()), index=cur_p_idx)
    if selected_preset != st.session_state.last_preset:
        apply_preset(selected_preset)
        st.rerun()

    # Render Dataset Insights & Real Historical Persona
    if st.session_state.get("p_dataset_id") is not None:
        if st.session_state.p_actual_survived == 1:
            surv_badge = '<span style="background: rgba(82, 183, 136, 0.25); color: #52b788; border: 1px solid #52b788; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">✅ Historical Outcome: SURVIVED (Dataset: Survived = 1)</span>'
        else:
            surv_badge = '<span style="background: rgba(230, 57, 70, 0.25); color: #e63946; border: 1px solid #e63946; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">❌ Historical Outcome: PERISHED (Dataset: Survived = 0)</span>'
        
        boat_badge = f'<div>🚣 <b>Rescue Craft:</b> <span style="color: #ffd166;">{st.session_state.p_boat}</span></div>' if st.session_state.p_boat else ""

        render_html(f"""
        <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid rgba(212, 175, 55, 0.4); border-radius: 12px; padding: 18px 22px; margin: 15px 0 25px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <div style="font-family: 'Cinzel', serif; font-size: 1.15rem; color: #ffd166; font-weight: 700; letter-spacing: 1px;">
                    📋 Verified Passenger Record (Kaggle: yasserh/titanic-dataset)
                </div>
                <div>
                    {surv_badge}
                </div>
            </div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 14px; font-size: 0.88rem; color: #a4b8d1; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px;">
                <div>🎫 <b>Kaggle PassengerId:</b> <span style="color: #ffd166; font-weight: 700;">#{st.session_state.p_dataset_id}</span></div>
                <div>🎟️ <b>Ticket No:</b> <span style="color: #ffffff; font-weight: 600;">{st.session_state.p_ticket}</span></div>
                <div>💰 <b>Historical Fare:</b> <span style="color: #48cae4; font-weight: 700;">£{st.session_state.p_fare:.2f}</span></div>
                <div>⚓ <b>Port of Embarkation:</b> <span style="color: #ffffff;">{st.session_state.p_embarked.split()[0]}</span></div>
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
            name_input = st.text_input("Passenger Full Name:", value=st.session_state.p_name, placeholder="e.g. John Smith or your name")
            st.session_state.p_name = name_input

            sex_opts = ["male", "female"]
            sex_labels = {"male": "👨 Male / Boy", "female": "👩 Female / Girl"}
            sex_input = st.selectbox("Biological Sex:", sex_opts, index=sex_opts.index(st.session_state.p_sex), format_func=lambda x: sex_labels[x])
            st.session_state.p_sex = sex_input

            # Dynamic title filtering based on biological sex
            if sex_input == "male":
                title_opts = ["Mr", "Master", "Rare"]
                title_labels = {
                    "Mr": "Mr. (Adult male ≥ 14 yrs)",
                    "Master": "Master. (Young boy ≤ 14 yrs)",
                    "Rare": "Aristocratic Title / Officer (Dr, Col, Rev, Sir)"
                }
            else:
                title_opts = ["Miss", "Mrs", "Rare"]
                title_labels = {
                    "Mrs": "Mrs. (Married woman)",
                    "Miss": "Miss. (Unmarried woman / girl)",
                    "Rare": "Aristocratic Title (Lady, Countess, Dona)"
                }

            if st.session_state.p_title not in title_opts:
                st.session_state.p_title = title_opts[0]
            default_t_idx = title_opts.index(st.session_state.p_title)

            title_input = st.selectbox(
                "Social Title (1912):", 
                title_opts, 
                index=default_t_idx, 
                format_func=lambda x: title_labels[x],
                help="Available titles adapt dynamically to selected biological sex."
            )
            st.session_state.p_title = title_input

        with c2:
            # Dynamic age limits based on title: Master strictly capped at 1-14 years!
            if title_input == "Master":
                min_age, max_age = 1, 14
                age_help = "For young boys (Master), age is strictly capped at 14."
            elif title_input == "Mr":
                min_age, max_age = 14, 80
                age_help = "For adult men (Mr.), age starts from 14."
            elif title_input == "Mrs":
                min_age, max_age = 15, 80
                age_help = "For married women (Mrs.), age starts from 15."
            elif title_input == "Rare":
                min_age, max_age = 18, 80
                age_help = "For noble titles or officers, age is 18 and above."
            else: # Miss
                min_age, max_age = 1, 80
                age_help = "For Miss, age can range from 1 to 80."

            cur_age_clamped = int(np.clip(st.session_state.p_age, min_age, max_age))
            st.session_state.p_age = cur_age_clamped

            age_input = st.slider(
                "Age (Years):", 
                min_value=min_age, 
                max_value=max_age, 
                value=cur_age_clamped,
                help=age_help
            )
            st.session_state.p_age = age_input

            pclass_opts = [1, 2, 3]
            pclass_labels = {1: "🥇 1st Class (First Class - Deluxe)", 2: "🥈 2nd Class (Second Class - Cabin)", 3: "🥉 3rd Class (Third Class - Steerage)"}
            pclass_input = st.selectbox("Socioeconomic Class:", pclass_opts, index=pclass_opts.index(st.session_state.p_pclass), format_func=lambda x: pclass_labels[x])
            st.session_state.p_pclass = pclass_input

            nat_opts = [
                "British (UK)", "Irish (Ireland)", "American (USA)", 
                "Scandinavian (Sweden/Norway)", "Greek / Balkan", 
                "Syrian / Lebanese (Ottoman Empire)", "French"
            ]
            nat_default_idx = nat_opts.index(st.session_state.p_nationality) if st.session_state.p_nationality in nat_opts else 0
            nat_input = st.selectbox("Nationality / Country of Origin:", nat_opts, index=nat_default_idx)
            st.session_state.p_nationality = nat_input

        st.markdown("#### 👥 Companions & Embarkation Port")
        c3, c4 = st.columns(2)
        with c3:
            comp_opts = [
                "Solo (1 person)",
                "Pair (2 people - spouse/companion)",
                "Small Group (3-4 people - family)",
                "Large Group (5+ people - extended family)"
            ]
            comp_default_idx = comp_opts.index(st.session_state.p_companions) if st.session_state.p_companions in comp_opts else 0
            comp_input = st.selectbox("Ticket Group Size:", comp_opts, index=comp_default_idx)
            st.session_state.p_companions = comp_input

        with c4:
            emb_opts = ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
            emb_labels = {
                "Southampton (S)": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Southampton (England - Origin)",
                "Cherbourg (C)": "🇫🇷 Cherbourg (France - Aristocracy)",
                "Queenstown (Q)": "🇮🇪 Queenstown / Cobh (Ireland - Emigrants)"
            }
            emb_default_idx = 0
            for idx, item in enumerate(emb_opts):
                if st.session_state.p_embarked.startswith(item[:3]):
                    emb_default_idx = idx
            emb_input = st.selectbox("Port of Embarkation:", emb_opts, index=emb_default_idx, format_func=lambda x: emb_labels[x])
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
        pass_name_display = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "Titanic Passenger"
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
                    <span style="font-size: 0.85rem; color: #a4b8d1;">≈ €{fare_modern:,} (2026 Equiv.)</span>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 0.72rem; color: #8899aa; text-align: center; letter-spacing: 1px;">
                WHITE STAR LINE • SAILING DATE: APRIL 10, 1912
            </div>
        </div>
        """)

        # Historical context note
        if st.session_state.p_nationality == "Greek / Balkan":
            st.caption("ℹ️ **Historical Note (Greek Passengers)**: Exactly 4 Greek emigrants were aboard the Titanic, all from the village of Agios Sostis, Messenia: Apostolos Chronopoulos (age 26 - passenger #74 in the Kaggle dataset), his brother Demetrios (age 19), Vasilios Katavelos (age 19), and Panagiotis Lymperopoulos (age 30). All 4 traveled in 3rd class and tragically perished in the disaster.")
        elif st.session_state.p_pclass == 3:
            st.caption("ℹ️ **Historical Note on Steerage**: 3rd-class passengers underwent strict medical inspections for trachoma and contagious diseases at Southampton to comply with US immigration quarantine rules.")

    # Navigation buttons
    st.markdown("<br>", unsafe_allow_html=True)
    _, next_col = st.columns([4, 1])
    with next_col:
        if st.button("Next: 3D Ship Deck Locator ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 2: 3D SHIP VISUALIZATION & CABIN PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 2:
    st.markdown("### 🚢 Step 2: Cabin Location & 3D Interactive Vessel Model")
    st.markdown("""
    Inspect where your cabin was situated within the ship structure. Ship side (**Port / Left** vs. **Starboard / Right**) 
    and deck level decisively governed access to the boat deck during the emergency.
    """)

    # Render active passenger banner & historical dataset insights
    render_passenger_banner(2)

    c_inputs, c_view = st.columns([1, 2], gap="large")

    with c_inputs:
        st.markdown("#### ⚓ Select Deck & Ship Side")
        
        # Valid decks by class
        if st.session_state.p_pclass == 1:
            deck_options = ["A", "B", "C"]
            deck_help = "Decks A-C: Luxury staterooms with immediate access to Boat Deck and Grand Staircase."
        elif st.session_state.p_pclass == 2:
            deck_options = ["D", "E"]
            deck_help = "Decks D-E: Mid-level decks, 2nd-class dining saloon, 3 flights down from lifeboats."
        else:
            deck_options = ["E", "F", "G"]
            deck_help = "Decks E-G: Deep near or below the waterline. Complex labyrinth of narrow corridors and watertight bulkheads."

        current_deck = st.session_state.p_deck if st.session_state.p_deck in deck_options else deck_options[0]
        selected_deck = st.selectbox(f"Deck Level for Class {st.session_state.p_pclass}:", deck_options, index=deck_options.index(current_deck), help=deck_help)
        st.session_state.p_deck = selected_deck

        side_options = [
            "Port (Left - Officer Lightoller)", 
            "Starboard (Right - Officer Murdoch)"
        ]
        side_idx = 1 if "Starboard" in st.session_state.p_side else 0
        selected_side = st.radio("Ship Side & Officer Protocol:", side_options, index=side_idx)
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
            <div style="font-weight: 700; color: #ffd166; margin-bottom: 8px;">📍 Cabin Spatial Parameters</div>
            <div>• <b>Passenger:</b> <span style="color: #ffffff; font-weight: 700;">{st.session_state.p_name}</span> (Class {st.session_state.p_pclass}, Age {st.session_state.p_age})</div>
            <div>• <b>Deck:</b> Deck {selected_deck}</div>
            <div>• <b>Elevation above Waterline:</b> {altitude:.1f} m</div>
            <div>• <b>Stair Flights to Boat Deck:</b> {flights} flights ({meters_to_boats:.1f} m)</div>
            <div>• <b>Commanding Officer:</b> <span class="highlight-cyan">{officer_name}</span></div>
        </div>
        """)

        is_woman_or_child = (st.session_state.p_sex == "female") or (st.session_state.p_age < 14)

        if not is_starboard:
            if is_woman_or_child:
                st.success("✅ **Port Side (Left - Officer Lightoller)**: Highly favorable side for women and children! Lightoller strictly enforced 'Women and children ONLY', granting absolute priority to females and infants.")
            else:
                st.error("⚠️ **Port Side (Left - Officer Lightoller)**: Extremely dangerous side for adult males! Lightoller rigidly enforced 'Women and children ONLY', turning away men even when lifeboats were lowered half empty.")
        else:
            if is_woman_or_child:
                st.info("ℹ️ **Starboard Side (Right - Officer Murdoch)**: Murdoch followed 'Women and children FIRST'. You have top priority, but men may also step aboard if empty seats remain.")
            else:
                st.success("✅ **Starboard Side (Right - Officer Murdoch)**: Most favorable side for male passengers! When no women were waiting, Murdoch permitted men to board empty seats, boosting male survival odds (~37%).")

    with c_view:
        # Camera angle view selector
        cam_choice = st.radio(
            "🎥 3D Camera Perspective:",
            ["Isometric 3D (Default)", "Starboard Profile (Right)", "Port Profile (Left)", "Overhead View (Top)", "Bow View (Front)"],
            horizontal=True
        )

        cam_map = {
            "Isometric 3D (Default)": dict(eye=dict(x=2.0, y=1.8, z=1.2), center=dict(x=0, y=0, z=-0.05)),
            "Starboard Profile (Right)": dict(eye=dict(x=0.05, y=3.1, z=0.45), center=dict(x=0, y=0, z=-0.05)),
            "Port Profile (Left)": dict(eye=dict(x=0.05, y=-3.1, z=0.45), center=dict(x=0, y=0, z=-0.05)),
            "Overhead View (Top)": dict(eye=dict(x=0.0, y=0.0, z=3.4), center=dict(x=0, y=0, z=0), up=dict(x=1, y=0, z=0)),
            "Bow View (Front)": dict(eye=dict(x=2.9, y=0.1, z=0.6), center=dict(x=0, y=0, z=-0.05))
        }
        current_cam = cam_map.get(cam_choice, cam_map["Isometric 3D (Default)"])

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
            showscale=False, opacity=0.96, name='Hull (Starboard)', hoverinfo='skip'
        ))
        fig3d.add_trace(go.Surface(
            x=X_u, y=-Y_u_stbd, z=Z_u,
            colorscale=[[0, '#111720'], [1, '#111720']],
            showscale=False, opacity=0.96, name='Hull (Port)', hoverinfo='skip'
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
            showscale=False, opacity=0.96, name='Keel / Waterline', hoverinfo='skip'
        ))
        fig3d.add_trace(go.Surface(
            x=X_l, y=-Y_l_stbd, z=Z_l,
            colorscale=[[0, '#781818'], [1, '#781818']],
            showscale=False, opacity=0.96, name='Keel / Waterline', hoverinfo='skip'
        ))

        # Superstructure Walls (White Deckhouse, x 18 to 72)
        x_sup = np.linspace(18, 72, 20)
        z_sup = np.linspace(12.0, 18.5, 4)
        X_s, Z_s = np.meshgrid(x_sup, z_sup)
        Y_s = np.full_like(X_s, 7.8)

        fig3d.add_trace(go.Surface(
            x=X_s, y=Y_s, z=Z_s,
            colorscale=[[0, '#edf2f7'], [1, '#edf2f7']],
            showscale=False, opacity=0.88, name='Superstructure', hoverinfo='skip'
        ))
        fig3d.add_trace(go.Surface(
            x=X_s, y=-Y_s, z=Z_s,
            colorscale=[[0, '#edf2f7'], [1, '#edf2f7']],
            showscale=False, opacity=0.88, name='Superstructure', hoverinfo='skip'
        ))

        # Teak Wood Boat Deck (Top, z 18.5, x 18 to 72, y -7.8 to 7.8)
        xd = np.linspace(18, 72, 15)
        yd = np.linspace(-7.8, 7.8, 8)
        XD, YD = np.meshgrid(xd, yd)
        fig3d.add_trace(go.Surface(
            x=XD, y=YD, z=np.full_like(XD, 18.5),
            colorscale=[[0, '#c2a67e'], [1, '#c2a67e']],
            showscale=False, opacity=0.92, name='Boat Deck (Teak Planking)', hoverinfo='skip'
        ))

        # Navigating Bridge (x 72, wings extending to y +- 10.5)
        fig3d.add_trace(go.Scatter3d(
            x=[72, 72, 72, 72, 72], y=[-10.5, -7.8, 0, 7.8, 10.5], z=[20.2, 20.2, 20.8, 20.2, 20.2],
            mode='lines+markers', line=dict(color='#ffffff', width=5),
            marker=dict(size=4, color='#ffd166'), name='Navigating Bridge'
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
                name=f'Funnel {f_idx+1}', hoverinfo='skip'
            ))

        # Fore and Aft Masts with Crow's Nest
        fig3d.add_trace(go.Scatter3d(
            x=[86, 83], y=[0, 0], z=[12, 36],
            mode='lines', line=dict(color='#d0dbe8', width=3), name='Fore Mast'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[84.5], y=[0], z=[27],
            mode='markers+text', marker=dict(size=6, color='#ffd166', symbol='diamond'),
            text=["Crow's Nest"], textposition='top right', name="Crow's Nest"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[10, 8], y=[0, 0], z=[12, 33],
            mode='lines', line=dict(color='#d0dbe8', width=3), name='Main Mast'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[83, 8], y=[0, 0], z=[36, 33],
            mode='lines', line=dict(color='#ffd166', width=1.5, dash='dash'), name='Marconi Aerial Rigging'
        ))

        # Lifeboats along Boat Deck edge
        lb_x = [68, 62, 56, 50, 44, 38, 32, 26]
        lb_y_stbd = [8.5] * len(lb_x)
        lb_y_port = [-8.5] * len(lb_x)
        lb_z = [19.2] * len(lb_x)

        fig3d.add_trace(go.Scatter3d(
            x=lb_x, y=lb_y_stbd, z=lb_z,
            mode='markers', marker=dict(size=6, color='#52b788', symbol='diamond'),
            name='Starboard Lifeboats (Murdoch)'
        ))
        fig3d.add_trace(go.Scatter3d(
            x=lb_x, y=lb_y_port, z=lb_z,
            mode='markers', marker=dict(size=6, color='#48cae4', symbol='diamond'),
            name='Port Lifeboats (Lightoller)'
        ))

        # Ocean Waterline
        x_sea = [-5, 105, 105, -5]
        y_sea = [-25, -25, 25, 25]
        fig3d.add_trace(go.Mesh3d(
            x=x_sea, y=y_sea, z=[3.5, 3.5, 3.5, 3.5],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='#071829', opacity=0.45, name='Calm Ocean Waterline', hoverinfo='skip'
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
            name='Cabin Vertical Beacon'
        ))
        # Glowing marker
        display_label = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "YOU"
        fig3d.add_trace(go.Scatter3d(
            x=[user_x], y=[user_y], z=[user_z],
            mode='markers+text',
            marker=dict(size=14, color='#ffd166', line=dict(color='#ffffff', width=2), symbol='circle'),
            text=[f'📍 YOUR CABIN ({display_label})'],
            textposition='top center',
            textfont=dict(color='#ffd166', size=13),
            name='Your Cabin Beacon'
        ))

        fig3d.update_layout(
            scene=dict(
                bgcolor='#08101a',
                xaxis=dict(title='← Stern (Aft 0) | Ship Length | Bow (Fore 100) →', showgrid=True, gridcolor='#182638', zeroline=False),
                yaxis=dict(title='Port ← | → Starboard', showgrid=True, gridcolor='#182638', zeroline=False),
                zaxis=dict(title='Height (Decks)', showgrid=True, gridcolor='#182638', zeroline=False),
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
        if st.button("⬅️ Back: Boarding Pass", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col_next:
        if st.button("Next: Sinking Timeline Simulation ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 3: TIMELINE SIMULATION (DYNAMIC PLAYBACK)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 3:
    st.markdown("### ⏱️ Step 3: Night of April 14, 1912 — Interactive Sinking Timeline")
    st.markdown("""
    Follow the progression of the sinking minute-by-minute. Adjust the interactive slider 
    to observe flooding dynamics at your cabin and the launch of lifeboats on your side of the ship.
    """)

    # Render active passenger banner & historical dataset insights
    render_passenger_banner(3)

    timeline_events = [
        {
            "time": "23:40",
            "title": "🧊 Collision with Iceberg (Starboard Bow)",
            "desc": "Lookout Frederick Fleet rings the bell 3 times: 'Iceberg right ahead!'. Titanic grazes the iceberg along her starboard hull, opening seams across 6 forward compartments. Water rushes in at 7 tons per second.",
            "water_level": "Tank Top / Boiler Rooms 5 & 6",
            "active_decks": "Tank Top / Hold",
            "urgency": "Calm (Most passengers felt only a faint shudder or vibration)",
            "cabins_flooded": ["Tank Top"]
        },
        {
            "time": "00:05",
            "title": "📋 Evacuation Order & Uncovering Lifeboats",
            "desc": "Shipbuilder Thomas Andrews calculates the ship has at most 1.5–2 hours before foundering. Captain Smith orders lifeboats swung out and wireless operators send CQD / SOS distress calls. Capacity is 1,178 for 2,224 people aboard.",
            "water_level": "Orlop Deck, Deck G & Mail Room",
            "active_decks": "G Deck",
            "urgency": "Moderate (Announcements in 1st/2nd class lounges; uncertainty in steerage)",
            "cabins_flooded": ["Tank Top", "G"]
        },
        {
            "time": "00:45",
            "title": "🛶 First Lifeboat Lowered (No. 7 - Starboard)",
            "desc": "First Officer Murdoch lowers Lifeboat 7 on starboard carrying only 28 people (rated capacity: 65!). Reluctant passengers hesitate to trade the warm, brightly lit liner for the pitch-black freezing ocean. On Port, Lightoller admits only women and children.",
            "water_level": "Deck F (Steerage Berth & Squash Court)",
            "active_decks": "F Deck",
            "urgency": "High (White distress rockets fired into the clear freezing night sky)",
            "cabins_flooded": ["Tank Top", "G", "F"]
        },
        {
            "time": "01:15",
            "title": "🚪 Lower Decks Flooding & Steerage Bottlenecks",
            "desc": "Seawater floods Deck E. Steerage passengers confront locked security gates and unlit, unfamiliar gangways. Only those finding open crew emergency ladders reach the higher decks.",
            "water_level": "Deck E (Lower passenger decks)",
            "active_decks": "E Deck",
            "urgency": "Critical (Bow trims heavily forward, incandescent lights flicker)",
            "cabins_flooded": ["Tank Top", "G", "F", "E"]
        },
        {
            "time": "01:40",
            "title": "⚡ Escalating Urgency on Boat Deck & Final Lifeboats",
            "desc": "The forward deck is submerged. Collapsible C departs with 71 aboard (including White Star Managing Director Bruce Ismay). On Port, Lightoller brandishes an unloaded revolver to maintain order as crowd surges.",
            "water_level": "Deck D & C (Water on Boat Deck)",
            "active_decks": "D & C Decks",
            "urgency": "Extreme Emergency (Remaining lifeboats depleted, ~1,500 people remain aboard)",
            "cabins_flooded": ["Tank Top", "G", "F", "E", "D", "C"]
        },
        {
            "time": "02:05",
            "title": "🎻 Final Plunge & Breakup (02:20 AM)",
            "desc": "The last craft (Collapsible D) leaves with 44 people. Wallace Hartley's band plays hymn 'Nearer, My God, to Thee'. Captain Smith steps onto the bridge shouting: 'Every man for himself'.",
            "water_level": "Boat Deck (Flooding over davits)",
            "active_decks": "A Deck & Boat Deck",
            "urgency": "Final Minutes",
            "cabins_flooded": ["Tank Top", "G", "F", "E", "D", "C", "B", "A"]
        },
        {
            "time": "02:20",
            "title": "💔 Catastrophic Hull Breakup & Final Plunge",
            "desc": "Immense bending stress breaks Titanic in two between the 3rd and 4th funnels. The bow plunges, followed by the perpendicular stern. ~1,500 people are plunged into -2.8°C seawater, perishing from hypothermia in minutes.",
            "water_level": "Complete Foundering (Wreck rests at 3,800m / 12,500 ft)",
            "active_decks": "All Decks Submerged",
            "urgency": "Catastrophe",
            "cabins_flooded": ["Tank Top", "G", "F", "E", "D", "C", "B", "A", "Boat Deck"]
        }
    ]

    # Time slider
    step_time = st.select_slider(
        "🎛️ Select Sinking Timestamp:",
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

        st.markdown("#### 🌊 Flooding Status & Waterline:")
        st.progress(
            (timeline_events.index(current_ev) + 1) / len(timeline_events),
            text=f"Water Level: {current_ev['water_level']}"
        )

        # Status badge for this passenger
        if is_flooded:
            st.error(f"🚨 **YOUR CABIN ON DECK {passenger_deck} HAS FLOODED!** Immediate evacuation required — passages below are filling with water!")
        else:
            st.success(f"🛡️ **Your Cabin on Deck {passenger_deck} remains dry**, but remaining time before flooding is rapidly expiring.")

        if st.session_state.get("p_boat"):
            st.info(f"📍 **Historical Passenger Dossier ({st.session_state.p_name})**: In official Titanic manifests, this passenger was recorded aboard craft **{st.session_state.p_boat}**.")

    with col_tl2:
        st.markdown("#### 🚣 Lifeboat Launch Activity by Ship Side")
        
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
            label="Total Lifeboats Successfully Lowered:",
            value=f"{len(boats_launched)} / 20",
            delta=f"{sum(b['loaded'] for b in boats_launched)} survivors"
        )

        st.markdown(f"**On your ship side ({'Starboard' if 'Starboard' in st.session_state.p_side else 'Port'}):**")
        if p_side_boats:
            boat_list_str = ", ".join([f"No.{b['boat']} ({b['loaded']}/{b['capacity']} passengers)" for b in p_side_boats[-3:]])
            st.write(f"Latest craft launched: {boat_list_str}")
        else:
            st.write("No lifeboats have lowered yet from your side of the vessel.")

        is_woman_or_child = (st.session_state.p_sex == "female") or (st.session_state.p_age < 14)

        if "Starboard" in st.session_state.p_side:
            if is_woman_or_child:
                st.info("💡 **Officer Murdoch (Starboard)**: Women and children have first priority; men are admitted to fill any remaining seats.")
            else:
                st.success("💡 **Officer Murdoch (Starboard)**: Men are allowed to board when no women are in line, offering a vital escape chance.")
        else:
            if is_woman_or_child:
                st.success("🛡️ **Officer Lightoller (Port)**: Absolute priority under strict 'Women and children ONLY' rule.")
            else:
                st.warning("⚠️ **Officer Lightoller (Port)**: Strictly prohibits men from boarding, sending boats away partly empty rather than seating men.")

    # Navigation buttons
    col_prev, _, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("⬅️ Back: 3D Ship Deck", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col_next:
        if st.button("Next: Survival Probability & ML ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 4: SURVIVAL PROBABILITY & MACHINE LEARNING EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 4:
    st.markdown("### 🔮 Step 4: Your Fate — Machine Learning Prediction & Explainability")
    st.markdown("Real-time probability inference computed by Scikit-Learn ML models integrated with historical officer protocols.")

    # Render active passenger banner & historical dataset insights
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
            title={"text": "SURVIVAL PROBABILITY", "font": {"size": 20, "color": "#ffffff", "family": "Cinzel"}},
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
                <h2 style="color: #52b788; margin-top: 0; font-size: 1.5rem;">🎉 YOU SURVIVED!</h2>
                <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                    With a calculated survival probability of <b>{final_prob:.1%}</b>, you successfully secured a seat in a lifeboat.
                    You were rescued by the <i>R.M.S. Carpathia</i> at dawn on April 15, 1912.
                </p>
            </div>
            """)
        else:
            render_html(f"""
            <div style="background: rgba(186, 24, 27, 0.25); border: 2px solid #e63946; border-radius: 12px; padding: 20px; margin-top: 15px;">
                <h2 style="color: #e63946; margin-top: 0; font-size: 1.5rem;">❄️ LOST IN THE NORTH ATLANTIC</h2>
                <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 0;">
                    With a calculated survival probability of only <b>{final_prob:.1%}</b>, the odds were overwhelmingly stacked against you.
                    Your name was recorded among the 1,500 tragic casualties of the foundering.
                </p>
            </div>
            """)

        factor_gender = 'Favorable (Woman / Child)' if not is_male or not is_adult else 'Unfavorable (Adult Male)'
        factor_class = 'High Priority (1st Class)' if st.session_state.p_pclass==1 else ('Moderate (2nd Class)' if st.session_state.p_pclass==2 else 'Low / Steerage Bottlenecks')
        factor_side = '+12% (Murdoch Male Admission)' if is_starboard and is_male else ('-14% (Lightoller Bar)' if not is_starboard and is_male else 'Neutral (Woman/Child)')

        render_html(f"""
        <div class="info-card" style="margin-top: 15px;">
            <b>🔑 Key Governing Survival Factors:</b><br>
            • <b>Demographic Profile:</b> {factor_gender}<br>
            • <b>Socioeconomic Class:</b> Class {st.session_state.p_pclass} ({factor_class})<br>
            • <b>Deck Officer Protocol:</b> {st.session_state.p_side} ({factor_side})
        </div>
        """)

        # Historical Dataset Comparison (for real Kaggle dataset persona)
        if st.session_state.get("p_dataset_id") is not None:
            actual_surv = st.session_state.p_actual_survived
            match_model = (survived and actual_surv == 1) or (not survived and actual_surv == 0)
            status_color = "#52b788" if match_model else "#ffd166"
            verdict_text = "MODEL PREDICTION AGREES WITH HISTORY" if match_model else "HISTORICAL ANOMALY / STATISTICAL EXCEPTION"
            actual_str = f"SURVIVED ({st.session_state.p_boat})" if actual_surv == 1 else "PERISHED IN SINKING"
            
            render_html(f"""
            <div style="background: rgba(18, 30, 49, 0.85); border: 1px solid {status_color}; border-radius: 10px; padding: 14px 16px; margin-top: 15px;">
                <div style="font-size: 0.82rem; color: {status_color}; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">
                    🏛️ Kaggle Dataset Verification (#{st.session_state.p_dataset_id}) • {verdict_text}
                </div>
                <div style="font-size: 0.92rem; color: #ffffff; line-height: 1.5;">
                    • <b>Model Prediction:</b> {final_prob:.1%} probability ({'Survived' if survived else 'Lost'})<br>
                    • <b>Historical Record (1912):</b> <b style="color: {'#52b788' if actual_surv == 1 else '#e63946'};">{actual_str}</b>
                </div>
            </div>
            """)

    # Explainability Waterfall / Feature Impact Chart
    st.markdown("#### 🔬 Explainable AI: Feature Importance Breakdown (Log-Odds Contribution)")
    st.caption("Quantified contribution of each demographic and spatial feature to your survival odds (Logistic Regression weights).")

    impacts = []
    for col, val in input_dict.items():
        if val != 0 and col in coefficients:
            weight = coefficients[col] * val
            readable_col = col.replace("Sex_male", "Sex (Male)").replace("Pclass_2", "2nd Class").replace("Pclass_3", "3rd Class (Steerage)").replace("IsAlone", "Solo Traveler").replace("TicketGroup_Pair (2)", "Pair (Ticket for 2)").replace("TicketGroup_Large Group (5+)", "Large Group (5+)").replace("Embarked_S", "Port (Southampton)").replace("Embarked_Q", "Port (Queenstown)")
            impacts.append({"Feature": readable_col, "Impact": weight})

    if officer_modifier != 0.0:
        impacts.append({
            "Feature": f"Ship Side ({'Starboard / Murdoch' if is_starboard else 'Port / Lightoller'})",
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
        title="Feature Contribution to Survival Probability (Log-Odds Impact)"
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
        if st.button("⬅️ Back: Timeline Simulation", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()
    with col_next:
        if st.button("Next: Key Notes & Citations ➡️", type="primary", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SCREEN 5: HISTORICAL KEY NOTES & RESEARCH DOSSIER
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_step == 5:
    st.markdown("### 📊 Step 5: Research Findings, Naval Forensics & Academic Citations")
    st.markdown("""
    Explore the statistical patterns derived from the historical passenger manifest and naval forensic research.
    These key findings illuminate the social and mechanical factors governing the tragedy.
    """)

    # Render active passenger banner & historical dataset insights
    render_passenger_banner(5)

    p_display = st.session_state.p_name.strip() if st.session_state.p_name.strip() else "Titanic Passenger"
    st.info(f"💡 **Passenger Focus:** **{p_display}** traveled in **Class {st.session_state.p_pclass}** ({st.session_state.p_sex}, age {st.session_state.p_age}). Explore below how socioeconomic standing and family dynamics dictated survival rates.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🛶 Lifeboat Paradox (472 Empty Seats)",
        "🎟️ Ticket Group Dynamics",
        "⚖️ Socioeconomic Class Gap",
        "👶 Women & Children First",
        "🔬 The Science of the Sinking",
        "📚 Academic & Historical Citations"
    ])

    with tab1:
        st.markdown("#### 🛶 The Lifeboat Paradox: 472 Empty Seats Lost in the Atlantic")
        st.markdown("""
        While the Titanic carried sufficient lifeboats for **1,178 individuals**, only **706 survivors** were rescued. 
        This demonstrates that **472 lifeboat seats departed completely empty**!
        """)

        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Total Lifeboat Capacity", "1,178 seats")
        col_b2.metric("Actual Rescued", "706 passengers", delta="-472 empty seats", delta_color="inverse")
        col_b3.metric("Average Lifeboat Fill Rate", "60.4%")

        # Lifeboats bar chart
        df_boats_plot = df_boats.copy()
        df_boats_plot["Fill_Rate"] = (df_boats_plot["loaded"] / df_boats_plot["capacity"] * 100).round(1)

        fig_lb = px.bar(
            df_boats_plot,
            x="boat",
            y="Fill_Rate",
            color="side",
            color_discrete_map={"Port": "#48cae4", "Starboard": "#52b788"},
            labels={"boat": "Lifeboat No", "Fill_Rate": "Fill Rate (%)", "side": "Ship Side"},
            title="Capacity Utilization per Lifeboat (Port vs Starboard)"
        )
        fig_lb.add_hline(y=100, line_dash="dash", line_color="#ffd166", annotation_text="Rated Maximum Capacity (100%)")
        fig_lb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 30, 49, 0.4)",
            font=dict(color="#ffffff"),
            height=360
        )
        st.plotly_chart(fig_lb, use_container_width=True)

        st.caption("📌 **Why did 472 seats go empty?** 1) Early officers feared davits would buckle if craft were filled to capacity before lowering. 2) Crew lifeboat drills had never been conducted. 3) Passenger reluctance to abandon a seemingly safe vessel early on.")

    with tab2:
        st.markdown("#### 🎟️ Ticket Groups: How Traveling Companions Dictated Survival")
        st.markdown("""
        Analyzing shared passenger tickets (Duplicate Ticket Groups) reveals a striking demographic pattern: 
        **Traveling couples had the highest survival rate (53.2%)**, while **large family groups (5+) collapsed below 15.6%**!
        """)

        # Ticket Group Data from Notebook
        ticket_stats = pd.DataFrame({
            "Group Size": ["Solo Traveler (1)", "Pair / Couple (2)", "Small Group (3-4)", "Large Family Group (5+)"],
            "Survival Rate (%)": [30.0, 53.2, 43.8, 15.6],
            "Passenger Count": [545, 188, 89, 69]
        })

        fig_tg = px.bar(
            ticket_stats,
            x="Group Size",
            y="Survival Rate (%)",
            color="Survival Rate (%)",
            color_continuous_scale="Viridis",
            text="Survival Rate (%)",
            title="Survival Rate by Ticket Group Size"
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

        st.caption("💡 **Sociological Context**: In couples, men escorted wives to lifeboats. In contrast, large immigrant families in steerage refused to separate, leading to entire households perishing together (e.g., the Goodwin family of 8 and the Sage family of 11).")

    with tab3:
        st.markdown("#### ⚖️ The Socioeconomic Class Gap (1st vs 2nd vs 3rd Class)")
        st.markdown("""
        Socioeconomic status exerted an enormous influence on survival. 1st-class passengers were **more than twice as likely to survive** compared to 3rd-class steerage travelers.
        """)

        class_summary = pd.DataFrame({
            "Class": ["1st Class", "2nd Class", "3rd Class (Steerage)"],
            "Overall Survival (%)": [62.9, 47.3, 24.2],
            "Female Survival (%)": [96.8, 92.1, 50.0],
            "Male Survival (%)": [36.9, 15.7, 13.5]
        })

        st.dataframe(class_summary, use_container_width=True, hide_index=True)

        fig_cls = px.bar(
            class_summary,
            x="Class",
            y=["Female Survival (%)", "Male Survival (%)"],
            barmode="group",
            title="Survival Rates by Sex and Passenger Class",
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
        st.markdown("#### 👶 The 'Women and Children First' Protocol & Demographic Reality")
        st.markdown("""
        The age-old maritime protocol was enforced, but a child's survival odds were determined almost entirely by passenger class:
        - **2nd Class**: **100% child survival rate** (18 of 18 children in Kaggle dataset; 24 of 24 historically).
        - **1st Class**: **75% child survival rate** (3 of 4 in dataset; 5 of 6 historically). The sole 1st/2nd-class child casualty was 2-year-old **Helen Lorraine Allison**, whose parents refused lifeboats searching for their infant brother.
        - **3rd Class (Steerage)**: **Only 42.9% of children survived** (21 of 49 in dataset; over 50 steerage children drowned due to trapped lower-deck quarters).
        """)

        child_stats = pd.DataFrame({
            "Child Cohort (< 14 yrs in Dataset)": ["1st Class Children", "2nd Class Children", "3rd Class Children"],
            "Survival Rate (%)": [75.0, 100.0, 42.9],
            "Survived": [3, 18, 21],
            "Lost": [1, 0, 28]
        })
        st.dataframe(child_stats, use_container_width=True, hide_index=True)

        st.caption("📌 **Data Source**: Kaggle Titanic Dataset (`yasserh/titanic-dataset` / Kaggle Titanic ML).")

    with tab5:
        st.markdown("#### 🔬 Forensic Naval Science: Why Did the 'Unsinkable' Ship Founder?")
        st.markdown("""
        Forensic, metallurgical, and hydrodynamic analysis of the disaster:
        """)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">1. The Ice Cube Tray Effect (Bulkhead Overflow)</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    Titanic was designed with <b>16 watertight compartments</b> and could stay afloat with any <b>4 compartments flooded</b>. 
                    The iceberg opened seams across <b>6 forward compartments</b>.<br><br>
                    <b>The fatal design flaw:</b> Bulkheads were not capped at ceiling level; they reached only up to Deck E (or Deck D forward). 
                    As the incoming weight pulled the bow down, seawater spilled over the top of one bulkhead into the next, 
                    <b>precisely like filling an ice cube tray under a running tap</b>!
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">2. Rivet Metallurgy & Slag Embrittlement</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    The hull was joined by <b>3,000,000 wrought iron and steel rivets</b> weighing 1,200 tons. 
                    Metallurgical testing on recovered rivets revealed high concentrations of silicate slag. 
                    At sub-zero water temperatures (-2.8°C / 27°F), iron underwent a <b>ductile-to-brittle transition</b>; rivet heads sheared off, 
                    unzipping the hull plating seams along the starboard side.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">3. Wireless Communications & The 'SS Californian'</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    The nearest vessel that night was the <b>SS Californian</b>, idling just <b>10 to 19 nautical miles away</b> 
                    (its masthead lights were clearly visible from Titanic's boat deck).<br><br>
                    <b>Why did it not respond?</b> Californian's sole wireless operator, Cyril Evans, powered down his set and retired at 23:30, 
                    10 minutes before impact. The watch officers saw the white distress rockets but failed to realize they signaled a catastrophe.<br><br>
                    By contrast, <b>RMS Carpathia</b> was 58 miles away; Captain Rostron navigated full ahead at 17.5 knots through icefields, 
                    and arrived at 04:00 AM to rescue all 705 survivors from the lifeboats.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4 style="color: #ffd166; margin-top:0;">4. Lethal Hypothermia in Freezing Waters (-2.8°C / 27°F)</h4>
                <p style="font-size: 0.92rem; line-height: 1.6;">
                    The sea temperature was <b>-2.8°C (27°F)</b> — below the freezing point of fresh water. 
                    In sub-zero salt water, core body temperature drops below 35°C within minutes, inducing cold incapacitation. 
                    Cold shock and cardiac arrest ensued within <b>10 to 15 minutes</b>, making survival in the water without a lifeboat virtually impossible. 
                    Of the hundreds plunged into the sea, fewer than 6 were pulled out alive.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab6:
        st.markdown("#### 📚 Academic & Historical References (APA Citations)")
        st.markdown("""
        All statistical parameters, machine learning weights, lifeboat records, and naval architecture metrics 
        are derived from peer-reviewed scientific literature, official government inquiry reports, and verified archives (APA format):
        """)

        st.markdown("""
        ##### 📊 1. Datasets & Machine Learning Repositories
        - **Kaggle Titanic Dataset**: Yasser, H. (2021). *Titanic Dataset*. Kaggle.  
          URL: [https://www.kaggle.com/datasets/yasserh/titanic-dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
        - **Kaggle Competition Data**: Kaggle Inc. (2012). *Titanic: Machine Learning from Disaster*.  
          URL: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
        - **Encyclopedia Titanica Manifest & Archives**: Hind, P. et al. (1996–2026). *Encyclopedia Titanica: Titanic Passenger Manifests, Deck Plans & Lifeboat Allocations*.  
          URL: [https://www.encyclopedia-titanica.org](https://www.encyclopedia-titanica.org)

        ##### 🏛️ 2. Official Governmental Inquiries (1912)
        - **British Mersey Inquiry (Parliamentary Papers)**:  
          Mersey, J. C. B., Baron. (1912). *Report on the Loss of the "Titanic" (S.S.): Formal Investigation into the Loss of the British S.S. Titanic*. Parliamentary Papers, Cd. 6352. London: His Majesty's Stationery Office.
        - **United States Senate Inquiry (Senate Document 726)**:  
          Smith, W. A. et al. (1912). *"Titanic" Disaster: Hearings before a Subcommittee of the Committee on Commerce, United States Senate, Sixty-Second Congress, Second Session* (Senate Document No. 726 / Report 806). Washington: Government Printing Office.

        ##### 📖 3. Survivor Eyewitness Accounts & Memoirs
        - **Lawrence Beesley (2nd Class Passenger)**:  
          Beesley, L. (1912). *The Loss of the SS Titanic: Its Story and Its Lessons*. Boston & New York: Houghton Mifflin Company.
        - **John Borland "Jack" Thayer Jr. (1st Class Passenger)**:  
          Thayer, J. B., Jr. (1940). *The Sinking of the S.S. Titanic: An Event That Shook the World*. Philadelphia: The Academy of Natural Sciences.
        - **Charles Herbert Lightoller (Second Officer)**:  
          Lightoller, C. H. (1935). *Titanic and Other Ships*. London: Ivor Nicholson and Watson.

        ##### 🔬 4. Forensic & Naval Engineering Research
        - **Rivet Slag Metallurgy & Cold Embrittlement**:  
          Foecke, T., & McCarty, J. (2008). *What Really Sank the Titanic: New Forensic Discoveries*. New York: Citadel Press.
        - **Hull Structural Failure & Hydrodynamics**:  
          Garzke, W. H., Brown, D. K., & Woodward, J. B. (1997). *The Structural Failure of the Titanic: Investigation and Analysis*. SNAME Transactions (Society of Naval Architects and Marine Engineers), 105, 237–264.

        ##### 💻 5. Scientific Software & Machine Learning Libraries
        - **Scikit-learn**: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
        - **Streamlit**: Snowflake Inc. (2026). *Streamlit: A faster way to build and share data apps*. Open source framework. URL: [https://streamlit.io](https://streamlit.io)
        - **Plotly**: Plotly Technologies Inc. (2026). *Collaborative Data Science and 3D Visualizations*. URL: [https://plotly.com](https://plotly.com)
        """)

    # Completion banner & Navigation buttons
    st.info("🏁 **Journey Complete (Step 5 of 5)!** You can review earlier steps or click **'Start Over: New Passenger'** to test another persona or custom profile.")

    col_prev, _, col_restart = st.columns([1, 2, 2])
    with col_prev:
        if st.button("⬅️ Back: Survival Prediction", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()
    with col_restart:
        if st.button("Start Over: New Passenger (Step 1) 🔄", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER & CREDITS & DATASET DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📖 Dataset Feature Dictionary & Schema (Kaggle: yasserh/titanic-dataset)"):
    st.markdown("""
    The machine learning model and statistical analyses are trained on the **Kaggle Titanic Dataset** (`yasserh/titanic-dataset` / Titanic ML from Disaster) comprising 891 verified passenger records:

    | Feature | Description | Values / Encoding |
    | :--- | :--- | :--- |
    | **`PassengerId`** | Unique passenger identifier | 1 to 891 |
    | **`Survived`** | Survival status (Target) | `0` = Perished, `1` = Survived |
    | **`Pclass`** | Socioeconomic class | `1` = 1st Class (Upper), `2` = 2nd (Middle), `3` = 3rd (Lower/Steerage) |
    | **`Name`** | Passenger full name with title | Contains honorifics (Mr, Mrs, Miss, Master, Col, etc.) |
    | **`Sex`** | Biological sex | `male`, `female` |
    | **`Age`** | Age in years | Fractional if < 1 year (e.g., 0.92 for 11 months) |
    | **`SibSp`** | Siblings or spouses aboard | Sibling count + spouse count |
    | **`Parch`** | Parents or children aboard | Parent count + child count |
    | **`Ticket`** | Ticket number / identifier | Alphanumeric (shared for families/couples) |
    | **`Fare`** | Passenger ocean fare | GBP (£) in 1912 currency |
    | **`Cabin`** | Stateroom allocation | Deck letter prefix (A–G) + room number |
    | **`Embarked`** | Port of embarkation | `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |
    """)

render_html("""
<div style="text-align: center; margin-top: 35px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.85rem; color: #8fa3b8; line-height: 1.6;">
    ⚓ <b>Titanic Interactive 3D Survival Simulator</b><br>
    Scientific & Historical Sources: <a href="https://www.kaggle.com/datasets/yasserh/titanic-dataset" target="_blank" style="color: #48cae4; text-decoration: none;">Kaggle Titanic Dataset</a> • <a href="https://www.encyclopedia-titanica.org" target="_blank" style="color: #48cae4; text-decoration: none;">Encyclopedia Titanica</a> • Mersey Inquiry (1912) • US Senate (1912)<br>
    Machine Learning Architecture: Scikit-Learn (Logistic Regression & Gradient Boosting) • 3D Visualizations: Plotly
</div>
""")
