import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.express as px

# --- 1. SETUP & LOADING ---
st.set_page_config(page_title="Storm Tracker", layout="wide")

@st.cache_resource
def load_data_and_model():
    # Load Model
    model = joblib.load('storm_model.pkl')
    
    # LOAD YOUR DATASET HERE (Required for the plot!)
    df = pd.read_csv('ultimate_clean.csv') 

    def assign_region(row):
        lat = row['lat']
        lon = row['lng']
        
        if lat > 35 and lon < -90 and lon > -110:
            return 'CENTRAL'
        elif lat > 35 and lon > -90:
            return 'EAST'
        elif lat < 35:
            return 'SOUTH'
        elif lat > 35 and lon < -110:
            return 'WEST'
        else:
            return 'UNKNOWN'

# Apply the function to every row (axis=1)
    df['REGION'] = df.apply(assign_region, axis=1)

    dmg_model = joblib.load('damage_model.pkl')
    return model, df, dmg_model

try:
    model, df_fullclean, dmg_model = load_data_and_model()
except FileNotFoundError:
    st.error("Files not found. Make sure 'storm_model.pkl' and your CSV file are in the folder.")
    st.stop()

page = st.sidebar.selectbox("Choose a Page", ["🔮 Predictor", "📊 Analytics"])

#Page 1 - Predictor
if page == "🔮 Predictor":
    # 2. App Title and Description
    st.title("🌪️ Storm Type Predictor")
    st.write("Enter the location and date details below to predict the likely storm type.")

    # 3. Create Input Columns for a cleaner layout
    # --- 1. SETUP SESSION STATE ---
    # This ensures the variables exist before we try to use them
    if 'lat' not in st.session_state:
        st.session_state.lat = 35.0
    if 'lon' not in st.session_state:
        st.session_state.lon = -97.0

    # --- 2. CREATE THE INTERACTIVE MAP ---
    # Start the map at the current lat/lon
    US_BOUNDS = {
        "min_lat": 24.396308,  # Southern tip of Florida
        "max_lat": 49.384358,  # Northern border
        "min_lon": -125.0,     # West Coast
        "max_lon": -66.93457   # East Coast
    }

    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=4, min_zoom=4, min_lat=US_BOUNDS["min_lat"],
        max_lat=US_BOUNDS["max_lat"],
        min_lon=US_BOUNDS["min_lon"],
        max_lon=US_BOUNDS["max_lon"],
        max_bounds=True)

    # Add a marker for the currently selected spot
    folium.Marker(
        [st.session_state.lat, st.session_state.lon], 
        popup="Selected Location", 
        tooltip="Click to change"
    ).add_to(m)

    # --- 3. DISPLAY MAP & CAPTURE CLICKS ---
    st.write("Click the map to select a location:")
    map_data = st_folium(m, height=300, width=700)

    # If the user clicked the map, update the session state variables
    if map_data['last_clicked']:
        st.session_state.lat = map_data['last_clicked']['lat']
        st.session_state.lon = map_data['last_clicked']['lng']
        # Rerun the app immediately to update the input boxes below
        st.rerun() 

    # --- 4. SHOW THE INPUTS (Synced with Map) ---
    # Notice we use 'key' to link these widgets to the session state
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Location")
        # These boxes will now automatically update when the map is clicked!
        user_lat = st.number_input("Latitude", key='lat', format="%.4f")
        user_lon = st.number_input("Longitude", key='lon', format="%.4f")

    with col2:
        st.subheader("📅 Date")
        # (Your existing date code goes here)
        # 1. The user picks the month FIRST
        user_month = st.slider("Month", min_value=1, max_value=12, value=1)

        # 2. Logic to determine the maximum days in that specific month
        if user_month in [4, 6, 9, 11]:
            # April, June, September, November have 30 days
            max_days = 30
        elif user_month == 2:
            # February gets 29 days (allowing for leap-year predictions since we don't ask for a Year!)
            max_days = 29
        else:
            # All other months have 31 days
            max_days = 31

        # 3. Create the Day slider, dynamically locking its max_value to the variable we just created
        user_day = st.slider("Day", min_value=1, max_value=max_days, value=1)



    # 4. The Prediction Logic
    if st.button("Predict Storm Type", type="primary"):
        
        # Prepare input exactly how the model expects it: 2D array
        user_input = [[user_lat, user_lon, user_month, user_day]]
        
        try:
            # A. Get Prediction
            prediction = model.predict(user_input)[0]
            
            # B. Get Probabilities
            probabilities = model.predict_proba(user_input)[0]
            classes = model.classes_

            # 2. PREPARE DATA FOR THE DAMAGE MODEL
            # ---------------------------------------------------------
            # Ask the damage model exactly what columns it needs
            expected_cols = dmg_model.feature_names_in_
            
            # Create a 1-row DataFrame filled entirely with 0.0
            dmg_input_df = pd.DataFrame(np.zeros((1, len(expected_cols))), columns=expected_cols)
            
            # Fill in the 4 numerical columns 
            # (CRITICAL: Match the exact spelling from line 99 of your notebook)
            dmg_input_df.at[0, 'lat'] = user_lat
            dmg_input_df.at[0, 'lng'] = user_lon
            dmg_input_df.at[0, 'BEGIN_MONTH'] = user_month
            dmg_input_df.at[0, 'BEGIN_DAY'] = user_day
            
            # Figure out the name of the dummy column for the predicted storm
            # e.g., if storm_pred is "Tornado", the column is "EVENT_TYPE_Tornado"
            dummy_col_name = f'EVENT_TYPE_{prediction}'
            
            # Turn that specific column "on" by setting it to 1
            if dummy_col_name in dmg_input_df.columns:
                dmg_input_df.at[0, dummy_col_name] = 1
                
            # ---------------------------------------------------------
            # 3. PREDICT THE DAMAGE
            # ---------------------------------------------------------
            # Now the data perfectly matches what the model expects!
            damage_pred = dmg_model.predict(dmg_input_df)[0]
            
            # Display results
            no_storm_index = list(model.classes_).index('No Storm') 
            
            # Get the exact probability of Clear Skies
            clear_skies_prob = probabilities[no_storm_index]
            
            # "Any Storm" is just 100% minus the chance of Clear Skies
            storm_prob = max(probabilities)

            if storm_prob >= 0.30 and storm_prob != clear_skies_prob:
                st.error(f"🚨 **STORM WARNING:** The probability ({storm_prob*100:.1f}%) exceeds your threshold!")
                st.warning(f"**Estimated Property Damage:** ${damage_pred:,.2f}")
            # Add your damage model prediction here if a storm is triggered!
            
            
                        
                # Create a dataframe for the probabilities to chart them easily
                prob_df = pd.DataFrame({
                    "Storm Type": classes,
                    "Probability": probabilities
                }).set_index("Storm Type")

                filtered_prob_df = prob_df[prob_df["Probability"] >= 0.05]
                
                st.subheader("Probability Breakdown")
                
                # Display as a Bar Chart
                st.bar_chart(filtered_prob_df)
                
                # Optional: Display raw text data like your original script
                with st.expander("See detailed percentages"):
                    for storm_name, prob in zip(classes, probabilities):
                        if prob >= 0.05:
                            st.write(f"{storm_name}: **{prob:.1%}**")
            else:
                st.success(f"☀️ **CLEAR SKIES:** The probability ({storm_prob*100:.1f}%) did not trigger the warning.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# PAGE 2: ANALYTICS (The new plot) ---
elif page == "📊 Analytics":
    st.title("📈 Historical Storm Data")
    st.subheader("Distribution of storms by state in our dataset.")

    # -- PLOT ONE --
    
    # 1. Create the figure object explicitly
    fig = plt.figure(figsize=(15, 10))
    
    # 2. Draw the plot (using the loaded dataframe)
    ax = sns.countplot(
        y='state_name', 
        data=df_fullclean, 
        order=df_fullclean['state_name'].value_counts().index
    )
    
    ax.tick_params(axis='y', labelsize=9)
    ax.set_xlabel("Storms in 2025")
    
    # 3. Pass the figure to Streamlit to render it
    st.pyplot(fig)

    # -- PLOT TWO --
    st.subheader("Storms by Region")
    fig2 = plt.figure(figsize=(15, 10))

    ax = sns.countplot(y='EVENT_TYPE', data=df_fullclean, order = df_fullclean['EVENT_TYPE'].value_counts().iloc[:15].index, hue = 'REGION')
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel("Frequency")

    st.pyplot(fig2)

    # -- PLOT THREE --

    st.subheader("Accuracy Check")
    st.image("Confusion_Matrix_final.png", caption='pred vs actual confustion matrix', width=2000)


