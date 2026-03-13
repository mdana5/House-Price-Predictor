import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    /* Darken the background slightly to make white cards pop */
    .stApp {
        background-color: #f0f2f6;
    }

    /* Sidebar - Solid Navy Blue */
    [data-testid="stSidebar"] {
        background-color: #002366 !important;
    }

    /* FEATURE CARDS - This makes inputs highly visible */
    div[data-testid="column"] {
        background-color: #ffffff;
        border: 2px solid #002366; /* Bold Blue Border */
        padding: 25px;
        border-radius: 15px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* BOLD LABELS */
    .stMarkdown h3, label p {
        color: #002366 !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
    }

    /* BUTTON STYLING */
    .stButton>button {
        background-color: #002366;
        color: white;
        height: 60px;
        width: 100%;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        border: 4px solid #0056b3;
    }

    /* RESULT BOX */
    .result-banner {
        background-color: #002366;
        color: #ffffff;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 5px solid #0056b3;
    }
            /* FORCED VISIBILITY FOR NAVIGATION */
    [data-testid="stSidebarNav"] {
        background-color: transparent !important;
    }

    /* Make the Radio labels (the text) super bright and bold */
    [data-testid="stSidebar"] .st-bd, [data-testid="stSidebar"] .st-ae {
        color: #ffffff !important;
    }

   /* Target the text labels specifically inside the sidebar radio group */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        opacity: 1 !important; /* Prevents unselected options from fading */
    }

    /* Fix for newer Streamlit versions that use different span classes */
    [data-testid="stSidebar"] .st-ae div, [data-testid="stSidebar"] .st-af {
        color: white !important;
    }

    /* Ensure the radio button circles are also bright */
    [data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stWidgetLabel"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='color:white; text-align:center;'>🧭 NAVIGATION</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio(
        "SELECT VIEW:", 
        ["🏠 Prediction Tool", "⚙️ Model Settings"],
        key="nav_menu"
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; opacity:0.7; text-align:center;'></p>", unsafe_allow_html=True)
# Load Column Names
if not os.path.exists("pickle Files/Random Forest/model_columns.pkl"):
    st.error("Missing Model Files")
    st.stop()
model_columns = joblib.load("pickle Files/Random Forest/model_columns.pkl")

# --- PAGE 1: SETTINGS ---
if page == "⚙️ Model Settings":
    st.title("⚙️ Engine Configuration")
    st.session_state['model_choice'] = st.selectbox(
        "Select Active Model",
        ["Random Forest (Recommended)", "Linear Regression"]
    )
    st.info("Configuration saved for the Prediction Tool.")

# --- PAGE 2: PREDICTION TOOL ---
else:
    st.markdown("<h1 style='color:#002366;'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
    st.write("### Step 1: Define Property Features")

    # EACH COLUMN BELOW IS NOW A BOX (CARD)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📐 Space")
        area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
        basement = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

    with col2:
        st.markdown("### 🛏️ Rooms")
        bedrooms = st.number_input("Total Bedrooms", 0, 10, 3)
        bathrooms = st.number_input("Total Bathrooms", 0, 10, 2)

    with col3:
        st.markdown("### 🏗️ Quality")
        quality = st.select_slider("Build Quality (1-10)", list(range(1, 11)), 5)
        age = st.number_input("House Age (Years)", 0, 200, 10)

    # Location in its own large box
    st.markdown("### 📍 Location & Parking")
    c_loc1, c_loc2 = st.columns(2)
    with c_loc1:
        location_columns = [c for c in model_columns if c.startswith("Location_")]
        locations = [c.replace("Location_", "") for c in location_columns]
        location = st.selectbox("Select Neighborhood", locations)
    with c_loc2:
        garage = st.number_input("Garage Car Capacity", 0, 5, 2)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Large Prediction Button
    if st.button("🚀 RUN PREDICTION ENGINE"):
        # Load Model
        choice = st.session_state.get('model_choice', "Random Forest (Recommended)")
        if choice == "Random Forest (Recommended)":
            model = joblib.load("pickle Files/Random Forest/random_forest_model.pkl")
        else:
            model = joblib.load("pickle Files/Linear Regression/linear_regression_model.pkl")

        # Prepare Data
        input_data = {
            "GrLivArea": area, "BedroomAbvGr": bedrooms, "Age": age,
            "OverallQual": quality, "TotalBathrooms": bathrooms,
            "GarageCars": garage, "TotalBsmtSF": basement
        }
        for col in location_columns: input_data[col] = 0
        input_data["Location_" + location] = 1

        input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df)[0]

        # Massive High-Visibility Result
        st.markdown(f"""
        <div class="result-banner">
            <h2 style="color:white; margin:0;">ESTIMATED MARKET PRICE</h2>
            <h1 style="color:white; font-size:70px; margin:0;">$ {prediction:,.2f}</h1>
            <p style="color:#bdc3c7;">Calculated using {choice}</p>
        </div>
        """, unsafe_allow_html=True)

