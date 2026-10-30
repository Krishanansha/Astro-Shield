import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ASTRO-TRACK", page_icon="🛰️", layout="wide")

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

# --- SOUND + WHATSAPP ---
def play_sound():
    html = """<audio autoplay><source src="https://cdn.pixabay.com/download/audio/2021/08/04/audio_0625c8b9d2.mp3" type="audio/mpeg"></audio><script>document.querySelector('audio')?.play().catch(()=>{});</script>"""
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
    if not csv_file: return None, "No CSV found"
    try:
        df = pd.read_csv(csv_file)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols: return None, "No numeric columns"
        dist_col = None
        for c in num_cols:
            if "miss" in c.lower() or "distance" in c.lower(): dist_col = c; break
        if not dist_col: dist_col = num_cols[0]
        df['is_close'] = (df[dist_col] < df[dist_col].quantile(0.2)).astype(int)
        feature_cols = [c for c in num_cols if c != 'is_close'][:6]
        if len(feature_cols) < 2: feature_cols = num_cols[:4]
        X = df[feature_cols].fillna(0); y = df['is_close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train); acc = model.score(X_test, y_test)
        return (model, feature_cols, csv_file, acc, df), None
    except Exception as e: return None, f"AI error: {e}"

model_info, model_err = train_ai()

# --- LIVE DATA ---
@st.cache_data(ttl=3600)
def get_live(days=2):
    today = datetime.utcnow().date()
    end = today + timedelta(days=days)
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={today}&end_date={end}&api_key={NASA_KEY}"
    r = requests.get(url, timeout=20); r.raise_for_status(); data = r.json()
    rows=[]
    for dk in data['near_earth_objects']:
        for neo in data['near_earth_objects'][dk]:
            ca = neo['close_approach_data'][0]
            rows.append({
                "id": neo['id'], "Name": neo['name'],
                "Diameter (m)": neo['estimated_diameter']['meters']['estimated_diameter_max'],
                "Distance_km": float(ca['miss_distance']['kilometers']),
                "Distance_LD": float(ca['miss_distance']['lunar']),
                "Distance_AU": float(ca['miss_distance']['astronomical']),
                "Distance_Miles": float(ca['miss_distance']['miles']),
                "Velocity_kms": float(ca['relative_velocity']['kilometers_per_second']),
                "Hazardous": neo['is_potentially_hazardous_asteroid'],
                "Date": dk, "JPL_URL": neo['nasa_jpl_url']
            })
    return pd.DataFrame(rows).sort_values("Distance_km")

@st.cache_data(ttl=3600)
def get_fireballs():
    url="https://ssd-api.jpl.nasa.gov/fireball.api?limit=200"
    r=requests.get(url, timeout=20); r.raise_for_status(); d=r.json()
    fixed=[]
    for row in d.get('data',[]):
        try:
            lat=float(row[0]); lat_dir=row[1]; lon=float(row[2]); lon_dir=row[3]
            if lat_dir=='S': lat=-lat
            if lon_dir=='W': lon=-lon
            fixed.append({"lat":lat,"lon":lon,"energy_kT":float(row[8]) if row[8] else 0,"date":row[5],"velocity":row[4]})
        except: continue
    return pd.DataFrame(fixed)

@st.cache_data(ttl=3600)
def get_sentry():
    url="https://ssd-api.jpl.nasa.gov/sentry.api"
    r=requests.get(url, timeout=20); r.raise_for_status(); d=r.json()
    rows=[]
    for obj in d.get('data',[]):
        try:
            rows.append({
                "Name": obj.get('des',''), "Diameter (m)": float(obj.get('diameter','0') or 0)*1000 if obj.get('diameter') else 0,
                "Impact_Prob": obj.get('ps_cum',''), "Torino": obj.get('ts_max',''),
                "Palermo": obj.get('ps_max',''), "Last_Obs": obj.get('last_obs',''),
                "n_imp": obj.get('n_imp',''), "Range": obj.get('range','')
            })
        except: continue
    return pd.DataFrame(rows)

# --- SIDEBAR + UNITS ---
st.sidebar.header("⚙️ Settings")
unit = st.sidebar.selectbox("Distance Unit", ["LD (Lunar Distance)", "km", "AU", "Miles"], index=0)
alert_km = st.sidebar.slider("Alert threshold (km)", 50000, 50000000, 7500000, step=50000)
st.sidebar.write(f"≈ {alert_km/384400:.1f} LD | {alert_km/149597870:.4f} AU")

def format_dist(row):
    if "LD" in unit: return f"{row['Distance_LD']:.2f} LD"
    elif unit=="km": return f"{row['Distance_km']:,.0f} km"
    elif unit=="AU": return f"{row['Distance_AU']:.4f} AU"
    else: return f"{row['Distance_Miles']:,.0f} mi"

st.sidebar.info("PHA = orbit <7.5M km (19.5 LD) + size>140m. Not impact today.")
if model_info: st.sidebar.success(f"AI Active: {model_info[2]}\nAcc: {model_info[3]:.1%} | Rows: {len(model_info[4])}")
else: st.sidebar.warning(model_err)
enable_whatsapp = st.sidebar.checkbox("WhatsApp Alert")
enable_sound = st.sidebar.checkbox("Sound Alert", value=True)

# --- METRICS ---
try: df_live = get_live(days=2)
except Exception as e: st.error(f"NASA API error: {e} - Check Secrets"); st.stop()

if not df_live.empty:
    c1,c2,c3,c4 = st.columns(4)
    closest_display = format_dist(df_live.iloc[0])
    c1.metric("Objects (2 days)", len(df_live))
    c2.metric("Hazardous", int(df_live['Hazardous'].sum()))
    c3.metric("Closest", closest_display)
    c4.metric("Fastest", f"{df_live['Velocity_kms'].max():.1f} km/s")

close_df = df_live[df_live['Distance_km'] < alert_km]
if not close_df.empty:
    st.error(f"🚨 {len(close_df)} object(s) closer than {alert_km:,} km")
    if enable_sound: play_sound(); st.toast("Dangerous approach!", icon="⚠️")
    if enable_whatsapp and send_whatsapp(f"ASTRO-SHIELD ALERT: {len(close_df)} asteroids within {alert_km} km"): st.sidebar.success("WhatsApp sent!")
else: st.success(f"✅ All clear - no objects within {alert_km:,} km")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📡 Live Monitor", "🤖 AI + Probability", "📜 Past Impacts Map", "☄️ 100-Year Risk (Sentry)", "📚 Past Data (CSV)"])

with tab1:
    left, right = st.columns([2,1])
    with left:
        if not df_live.empty:
            y_col = "Distance_km" if unit=="km" else "Distance_LD" if "LD" in unit else "Distance_AU" if unit=="AU" else "Distance_Miles"
            fig = px.scatter(df_live, x=y_col, y="Diameter (m)", size="Velocity_kms", color="Hazardous", hover_name="Name", log_y=True, title=f"Distance ({unit}) vs Size")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if not df_live.empty:
            df_live['Risk_Score'] = (1/df_live['Distance_LD']) * np.log1p(df_live['Diameter (m)'])
            df_live['Hit_Prob_%'] = (df_live['Risk_Score']/df_live['Risk_Score'].max()*100).round(1)
            df_live['Display_Dist'] = df_live.apply(format_dist, axis=1)
            fig2 = px.bar(df_live.sort_values("Hit_Prob_%", ascending=False).head(10), x="Hit_Prob_%", y="Name", orientation='h', color="Hazardous", title="Relative Closeness Score")
            st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df_live, use_container_width=True, column_config={"JPL_URL": st.column_config.LinkColumn("Details", display_text="Open JPL"), "Hit_Prob_%": st.column_config.ProgressColumn(min_value=0, max_value=100)}, hide_index=True)

with tab2:
    if model_info:
        st.write(f"Trained on `{model_info[2]}` | Features: {model_info[1]} | Rows: {len(model_info[4])}")
        st.write(f"Accuracy: **{model_info[3]:.2%}**")
        fig_imp = px.bar(x=model_info[1], y=model_info[0].feature_importances_, title="Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)
    else: st.info("Upload neo.csv")

with tab3:
    st.subheader("Atmospheric Fireballs - Real Past Impacts (Where it hit)")
    st.caption("Data: NASA CNEOS Fireball API - 200 latest atmospheric impacts. Size = energy in kT")
    try:
        df_fb = get_fireballs()
        if "energy_kT" not in df_fb.columns:
        df_fb["energy_kT"] = 1
        df_fb["energy_kT"] = df_fb["energy_kT"].fillna(0.1).replace(0, 0.1)
        st.map(df_fb, latitude="lat", longitude="lon")
        st.dataframe(df_fb.sort_values("energy_kT", ascending=False), use_container_width=True)
        if not df_fb.empty:
            fig_fb = px.scatter_geo(df_fb, lat="lat", lon="lon", size="energy_kT", hover_name="date", color="energy_kT", title="Past Impact Locations (Fireballs)")
            st.plotly_chart(fig_fb, use_container_width=True)
    except Exception as e: st.error(f"Fireball error: {e}")

with tab4:
    st.subheader("NASA Sentry - 100 Year Risk List")
    st.caption("Objects with non-zero impact probability next 100 years. Palermo & Torino scales.")
    try:
        df_sentry = get_sentry()
        if not df_sentry.empty:
            st.dataframe(df_sentry, use_container_width=True)
            st.warning("Sentry objects are tracked for future orbits. Probability is very low (<0.1%) for most.")
        else: st.info("No Sentry data or API down.")
    except Exception as e: st.error(f"Sentry error: {e}")

with tab5:
    st.subheader("Past Data from your CSV (No 10 limit - Full History)")
    if model_info:
        df_hist = model_info[4]
        st.write(f"Showing all {len(df_hist)} records from {model_info[2]}")
        # show filters
        search = st.text_input("Search Name/ID")
        if search: df_hist = df_hist[df_hist.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        st.dataframe(df_hist, use_container_width=True, height=600)
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered CSV", csv, "filtered.csv", "text/csv")
    else: st.info("Upload neo.csv to see past data")

st.divider()
st.markdown(f"**Units:** 1 LD = 384,400 km = 0.00257 AU = 238,855 miles. Selected: {unit}. PHA is orbit potential, not today's risk.")
