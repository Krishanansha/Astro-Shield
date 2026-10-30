import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ASTRO-SHIELD", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
.main {background:#0a0f1f;}
h1 {color:#6cf !important;}
.metric-card {background:#111a33; padding:15px; border-radius:12px; border:1px solid #1f2a4a;}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ ASTRO-SHIELD")
st.markdown("### AI-Powered Near-Earth Object Early Monitoring")
st.caption("Educational monitor | NASA NeoWs + Fireballs + Sentry | Not an official warning")
st.divider()

NASA_KEY = st.secrets.get("NASA_KEY", "DEMO_KEY")
if NASA_KEY == "DEMO_KEY":
    st.warning("Using DEMO_KEY - add NASA_KEY in Streamlit Secrets for full limit.")

# --- SOUND ---
def play_sound():
    html = """
    <audio autoplay><source src="https://cdn.pixabay.com/download/audio/2021/08/04/audio_0625c8b9d2.mp3" type="audio/mpeg"></audio>
    <script>document.querySelector('audio')?.play().catch(()=>{});</script>
    """
    st.components.v1.html(html, height=0)

def send_whatsapp(msg):
    try:
        if "CALLMEBOT_APIKEY" in st.secrets:
            phone = st.secrets.get("ALERT_TO","").replace("whatsapp:","").replace("+","")
            apikey = st.secrets["CALLMEBOT_APIKEY"]
            url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={requests.utils.quote(msg)}&apikey={apikey}"
            requests.get(url, timeout=10)
            return True
    except: pass
    return False

# --- AI TRAIN ---
@st.cache_resource
def train_ai():
    csv_file = None
    for f in ["neo.csv","nasa.csv","dataset.csv","neo_kaggle.csv"]:
        if os.path.exists(f): csv_file = f; break
    if not csv_file:
        for f in os.listdir("."):
            if f.endswith(".csv"): csv_file = f; break
    if not csv_file:
        return None, "No CSV found - upload neo.csv"

    try:
        df = pd.read_csv(csv_file)
        # create target: close approach = bottom 20% distance
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols: return None, "No numeric columns in CSV"
        
        # pick distance-like column
        dist_col = None
        for c in num_cols:
            if "miss" in c.lower() or "distance" in c.lower():
                dist_col = c; break
        if not dist_col: dist_col = num_cols[0]
        
        df['is_close'] = (df[dist_col] < df[dist_col].quantile(0.2)).astype(int)
        feature_cols = [c for c in num_cols if c != 'is_close'][:6]
        if len(feature_cols) < 2: feature_cols = num_cols[:4]
        
        X = df[feature_cols].fillna(0)
        y = df['is_close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return (model, feature_cols, csv_file, acc), None
    except Exception as e:
        return None, f"AI error: {e}"

model_info, model_err = train_ai()

@st.cache_data(ttl=3600)
def get_live():
    today = datetime.utcnow().date()
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={today}&end_date={today}&api_key={NASA_KEY}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows=[]
    for dk in data['near_earth_objects']:
        for neo in data['near_earth_objects'][dk]:
            ca = neo['close_approach_data'][0]
            rows.append({
                "id": neo['id'],
                "Name": neo['name'],
                "Diameter (m)": neo['estimated_diameter']['meters']['estimated_diameter_max'],
                "Distance_km": float(ca['miss_distance']['kilometers']),
                "Distance_LD": float(ca['miss_distance']['lunar']),
                "Velocity_kms": float(ca['relative_velocity']['kilometers_per_second']),
                "Hazardous": neo['is_potentially_hazardous_asteroid'],
                "JPL_URL": neo['nasa_jpl_url']
            })
    return pd.DataFrame(rows).sort_values("Distance_km")

@st.cache_data(ttl=3600)
def get_fireballs():
    url="https://ssd-api.jpl.nasa.gov/fireball.api?limit=100"
    r=requests.get(url, timeout=20)
    r.raise_for_status()
    d=r.json()
    fixed=[]
    for row in d.get('data',[]):
        try:
            lat=float(row[0]); lat_dir=row[1]; lon=float(row[2]); lon_dir=row[3]
            if lat_dir=='S': lat=-lat
            if lon_dir=='W': lon=-lon
            fixed.append({"lat":lat,"lon":lon,"energy":row[8],"date":row[5]})
        except: continue
    return pd.DataFrame(fixed)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")
alert_km = st.sidebar.slider("Alert threshold (km)", 50000, 10000000, 7500000, step=50000)
st.sidebar.write(f"≈ {alert_km/384400:.1f} Lunar Distances")
st.sidebar.info("PHA = orbit can come within 7.5M km (19.5 LD) + size>140m. Not impact today.")
if model_info:
    st.sidebar.success(f"AI Active: {model_info[2]}\nAcc: {model_info[3]:.1%}")
else:
    st.sidebar.warning(model_err)
enable_whatsapp = st.sidebar.checkbox("WhatsApp Alert")
enable_sound = st.sidebar.checkbox("Sound Alert", value=True)

# --- METRICS ---
try:
    df_live = get_live()
except Exception as e:
    st.error(f"NASA API error: {e}")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Objects Today", len(df_live))
c2.metric("Hazardous", int(df_live['Hazardous'].sum()) if not df_live.empty else 0)
c3.metric("Closest", f"{df_live['Distance_LD'].min():.2f} LD" if not df_live.empty else "N/A")
c4.metric("Fastest", f"{df_live['Velocity_kms'].max():.1f} km/s" if not df_live.empty else "N/A")

# --- ALERT ---
close_df = df_live[df_live['Distance_km'] < alert_km]
if not close_df.empty:
    st.error(f"🚨 {len(close_df)} object(s) closer than {alert_km:,} km")
    if enable_sound: 
        play_sound()
        st.toast("Dangerous approach!", icon="⚠️")
    if enable_whatsapp:
        if send_whatsapp(f"ASTRO-SHIELD ALERT: {len(close_df)} asteroids within {alert_km} km. Closest {close_df.iloc[0]['Name']} at {close_df.iloc[0]['Distance_LD']:.2f} LD"):
            st.sidebar.success("WhatsApp sent!")
else:
    st.success(f"✅ All clear - no objects within {alert_km:,} km")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📡 Live Monitor", "🤖 AI + Probability", "📜 Fireballs"])

with tab1:
    left, right = st.columns([2,1])
    with left:
        if not df_live.empty:
            fig = px.scatter(df_live, x="Distance_LD", y="Diameter (m)", size="Velocity_kms",
                             color="Hazardous", hover_name="Name", log_y=True,
                             title="Distance vs Size - Click for details")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if not df_live.empty:
            df_live['Risk_Score'] = (1/df_live['Distance_LD']) * np.log1p(df_live['Diameter (m)'])
            df_live['Hit_Prob_%'] = (df_live['Risk_Score']/df_live['Risk_Score'].max()*100).round(1)
            fig2 = px.bar(df_live.sort_values("Hit_Prob_%", ascending=False).head(10),
                          x="Hit_Prob_%", y="Name", orientation='h', color="Hazardous",
                          title="Relative Closeness Score")
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Live Table - Click JPL_URL for NASA details")
    # AI prediction
    try:
        if model_info and not df_live.empty:
            model, fcols, _, _ = model_info
            live_cols = [c for c in fcols if c in df_live.columns]
            X_live = df_live[live_cols] if live_cols else pd.DataFrame()
            if not X_live.empty:
                if hasattr(model, "predict_proba"):
                    df_live['AI_Anomaly'] = model.predict_proba(X_live)[:,1].round(2)
                else:
                    df_live['AI_Anomaly'] = model.predict(X_live)
    except:
        df_live['AI_Anomaly'] = 0

    st.dataframe(df_live, use_container_width=True,
                 column_config={
                     "JPL_URL": st.column_config.LinkColumn("Details", display_text="Open JPL"),
                     "Hit_Prob_%": st.column_config.ProgressColumn(min_value=0, max_value=100)
                 }, hide_index=True)

    sel = st.selectbox("Select asteroid for details:", df_live['Name'].tolist() if not df_live.empty else [])
    if sel:
        st.json(df_live[df_live['Name']==sel].iloc[0].to_dict())

with tab2:
    st.subheader("Kaggle AI Model")
    if model_info:
        st.write(f"Trained on `{model_info[2]}` | Features: {model_info[1]}")
        st.write(f"Model: RandomForest 100 trees | Task: Detect unusually close approaches")
        st.write(f"Accuracy: **{model_info[3]:.2%}**")
        fig_imp = px.bar(x=model_info[1], y=model_info[0].feature_importances_, title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Upload Kaggle CSV as neo.csv to activate AI. https://www.kaggle.com/datasets/sameepels/nasa-near-earth-objects")

with tab3:
    st.subheader("Atmospheric Fireballs (corrected lat/lon)")
    try:
        df_fb = get_fireballs()
        st.map(df_fb, latitude="lat", longitude="lon")
        st.dataframe(df_fb, use_container_width=True)
    except Exception as e:
        st.error(f"Fireball error: {e}")

st.divider()
st.markdown("**How to read:** 1 LD = 384,400 km. PHA is orbit potential, not today's risk. Hit Prob is relative closeness score, not real impact probability.")
