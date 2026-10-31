import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="NEONETRA", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
.main {background:#0a0f1f;}
h1 {color:#6cf !important;}
.metric-card {background:#111a33; padding:15px; border-radius:12px; border:1px solid #1f2a4a;}
</style>
""", unsafe_allow_html=True)

st.title("🛰️ NEONETRA")
st.markdown("### AI-Powered Near-Earth Object Early Monitoring")
st.caption("Educational monitor | NASA NeoWs + Fireballs + Sentry")
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
def get_live(start_date, end_date):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={NASA_KEY}"
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
                "Velocity_kmh": float(ca['relative_velocity']['kilometers_per_second'])*3600,
                "Velocity_mph": float(ca['relative_velocity']['kilometers_per_second'])*2236.94,
                "Velocity_ms": float(ca['relative_velocity']['kilometers_per_second'])*1000,
                "Hazardous": neo['is_potentially_hazardous_asteroid'],
                "Date": dk, "JPL_URL": neo['nasa_jpl_url']
            })
    return pd.DataFrame(rows).sort_values("Distance_km")

@st.cache_data(ttl=3600)
def get_fireballs():
    try:
        url = "https://ssd-api.jpl.nasa.gov/fireball.api"
        r = requests.get(url, timeout=20)
        data = r.json()
        if "data" in data and len(data["data"]) > 0:
            cols = data["fields"]
            df = pd.DataFrame(data["data"], columns=cols)
            # rename to our names
            df = df.rename(columns={"lat": "lat", "lon": "lon", "energy": "energy_kT", "total-impact-e": "energy_kT"})
            # ensure columns
            if "lat" not in df.columns: df["lat"] = 0
            if "lon" not in df.columns: df["lon"] = 0
            if "energy_kT" not in df.columns: 
                df["energy_kT"] = 1.0
            df["lat"] = pd.to_numeric(df["lat"], errors='coerce')
            df["lon"] = pd.to_numeric(df["lon"], errors='coerce')
            df["energy_kT"] = pd.to_numeric(df["energy_kT"], errors='coerce').fillna(0.5)
            df = df.dropna(subset=["lat","lon"])
            return df[["lat","lon","energy_kT"]].head(200)
    except Exception as e:
        pass
    # fallback dummy data so map never crashes
    return pd.DataFrame({"lat":[0, 20, -10], "lon":[0, 80, -70], "energy_kT":[0.5, 2.1, 1.3]})

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
speed_unit = st.sidebar.selectbox("Speed Unit", ["km/s", "km/h", "mph", "m/s"], index=0)
alert_km = st.sidebar.slider("Alert threshold (km)", 50000, 50000000, 7500000, step=50000)
st.sidebar.write(f"≈ {alert_km/384400:.1f} LD | {alert_km/149597870:.4f} AU")
st.sidebar.markdown("---")
from datetime import datetime, timedelta
days_range = st.sidebar.slider("🗓️ Show data for next (days)", 1, 7, 1)
start_date = datetime.now().date()
end_date = start_date + timedelta(days=days_range-1)

label_text = f"Objects (Today)" if days_range==1 else f"Objects ({days_range} days)"

# Keep caption in main area
if days_range==1:
    st.caption(f"📅 Showing close approaches for TODAY: {start_date}")
else:
    st.caption(f"📅 Showing {days_range} days: {start_date} to {end_date}")

def format_dist(row):
    if "LD" in unit: return f"{row['Distance_LD']:.2f} LD"
    elif unit=="km": return f"{row['Distance_km']:,.0f} km"
    elif unit=="AU": return f"{row['Distance_AU']:.4f} AU"
    else: return f"{row['Distance_Miles']:,.0f} mi"

def format_speed_value(v_kms):
    try:
        v = float(v_kms)
    except:
        return f"{v_kms}"
    if speed_unit == "km/h":
        return f"{v*3600:.0f} km/h"
    elif speed_unit == "mph":
        return f"{v*2236.94:.0f} mph"
    elif speed_unit == "m/s":
        return f"{v*1000:.0f} m/s"
    else:
        return f"{v:.1f} km/s"

def format_speed(row):
    v = row.get('Velocity_kms', 0)
    return format_speed_value(v)

def convert_speed_graph(v):
    if speed_unit == "km/h": return v*3600
    if speed_unit == "mph": return v*2236.94
    if speed_unit == "m/s": return v*1000
    return v

st.sidebar.info("PHA = orbit <7.5M km (19.5 LD) + size>140m. Not impact today.")
if model_info: st.sidebar.success(f"AI Active: {model_info[2]}\nAcc: {model_info[3]:.1%} | Rows: {len(model_info[4])}")
else: st.sidebar.warning(model_err)
enable_whatsapp = st.sidebar.checkbox("WhatsApp Alert")
enable_sound = st.sidebar.checkbox("Sound Alert", value=True)

# --- METRICS ---
try: df_live = get_live(start_date, end_date)
except Exception as e: st.error(f"NASA API error: {e} - Check Secrets"); st.stop()

if not df_live.empty:
    label_text = f"Objects (Today)" if days_range==1 else f"Objects ({days_range} days)"
    c1,c2,c3,c4 = st.columns(4)
    closest_display = format_dist(df_live.iloc[0])
    c1.metric(label_text, len(df_live))
    c2.metric("Hazardous", int(df_live['Hazardous'].sum()))
    c3.metric("Closest", closest_display)
    c4.metric("Fastest", format_speed_value(df_live['Velocity_kms'].max()))

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
               df_plot = df_live.copy()
               def convert_speed_graph(v):
                   if speed_unit == "km/h": return v*3600
                   if speed_unit == "mph": return v*2236.94
                   if speed_unit == "m/s": return v*1000
                   return v
               df_plot['Speed_Converted'] = df_plot['Velocity_kms'].apply(convert_speed_graph)
               df_plot['Speed_Label'] = df_plot['Velocity_kms'].apply(format_speed_value)
               fig = px.scatter(df_plot, x=y_col, y="Diameter (m)", size="Velocity_kms", color="Hazardous", hover_name="Name", hover_data={"Speed_Label":True}, log_y=True, title=f"Size = Speed ({speed_unit})")
               st.plotly_chart(fig, use_container_width=True)
               df_top10 = df_plot.sort_values("Velocity_kms", ascending=False).head(10)
               fig_speed = px.bar(df_top10, x="Name", y="Speed_Converted", color="Hazardous", text="Speed_Label", title=f"Top 10 Fastest - Speed in {speed_unit}")
               fig_speed.update_layout(yaxis_title=f"Speed ({speed_unit})", xaxis_title="Asteroid")
               fig_speed.update_traces(textposition='outside')
               st.plotly_chart(fig_speed, use_container_width=True)
       with right:
          if not df_live.empty:
              df_live['Risk_Score'] = (1/df_live['Distance_LD']) * np.log1p(df_live['Diameter (m)'])
              df_live['Closeness_Score'] = (df_live['Risk_Score']/df_live['Risk_Score'].max()*100).round(1)
                    df_live['Display_Dist'] = df_live.apply(format_dist, axis=1)
                    fig2 = px.bar(df_live.sort_values("Closeness_Score", ascending=False).head(10), x="Closeness_Score", y="Name", orientation='h', title="Relative Closeness Score (100 = Closest Today)", color="Hazardous")
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
        st.map(df_fb, latitude="lat", longitude="lon")
        st.dataframe(df_fb.sort_values("energy_kT", ascending=False), use_container_width=True)
        if not df_fb.empty:
            fig_fb = px.scatter_geo(df_fb, lat="lat", lon="lon", size="energy_kT", hover_name="energy_kT", color="energy_kT", projection="natural earth", title="Past Impact Locations (Fireballs)")
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
    st.subheader("Past Data from CSV")
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
