import streamlit as st
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Attempt to import Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None # Placeholder if not installed

# Enhanced page configuration
st.set_page_config(
    page_title="Environment Impact Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57 0%, #32CD32 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: black; /* Lighter background for cards */
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; color: #856404; }
    .alert-danger { background-color: #f8d7da; border-left: 4px solid #dc3545; color: #721c24; }
    .alert-success { background-color: #d4edda; border-left: 4px solid #28a745; color: #155724; }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #32CD32;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Enhanced title with styling
st.markdown("""
<div class="main-header">
    <h1>🌿 Environment Impact Analyzer</h1>
    <p>Advanced Environmental Data Analysis & Sustainability Planning</p>
</div>
""", unsafe_allow_html=True)

# --- Global NpEncoder Class ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj) if pd.notna(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj): # Catches pd.NA, np.nan
            return None
        if isinstance(obj, (datetime, pd.Timestamp)): # Handle datetime objects
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# --- Sample Data Generation ---
@st.cache_data
def get_sample_data():
    """Generates sample Pandas DataFrames for climate, air, soil, and water."""
    cities = [
        # Tamil Nadu Cities (Expanded)
    "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Erode",
    "Tirunelveli", "Tiruppur", "Vellore", "Thoothukudi", "Nagercoil", "Dindigul",
    "Thanjavur", "Karur", "Namakkal", "Kanchipuram", "Cuddalore", "Villupuram",
    "Nagapattinam", "Kanyakumari", "Ariyalur", "Perambalur", "Tiruvarur",
    "Sivaganga", "Virudhunagar", "Ramanathapuram", "Theni", "Krishnagiri",
    "Dharmapuri", "Nilgiris", "Pudukkottai", "Tenkasi", "Mayiladuthurai",
    "Ranipet", "Tirupattur", "Chengalpattu",

    # Other Indian Cities
    "Mumbai", "Delhi", "Bengaluru", "Kolkata", "Hyderabad", "Pune", "Ahmedabad",
    "Jaipur", "Lucknow", "Surat", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal",
    "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik",
    "Faridabad", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad",
    "Amritsar", "Navi Mumbai", "Allahabad", "Ranchi", "Howrah", "Jabalpur",
    "Gwalior", "Vijayawada", "Jodhpur", "Raipur", "Guwahati", "Chandigarh",
    "Solapur", "Mysore", "Gurgaon", "Kochi", "Jalandhar", "Bhubaneswar",
    "Noida", "Thiruvananthapuram", "Saharanpur", "Gorakhpur", "Guntur",
    "Shimla", "Dehradun"
    ]
    base_date = datetime(2023, 1, 1)
    num_days = 30 # Generate 30 days of data

    climate_records = []
    for city in cities:
        for i in range(num_days):
            date = (base_date + timedelta(days=i)).strftime('%Y-%m-%d')
            climate_records.append({
                'city': city, 'date': date,
                'temperature_mean': round(np.random.uniform(5, 40), 1),
                'temperature_max': round(np.random.uniform(10, 45), 1),
                'temperature_min': round(np.random.uniform(0, 35), 1),
                'wind_speed_mean': round(np.random.uniform(1, 25), 1),
                'wind_speed_max': round(np.random.uniform(3, 40), 1),
                'relative_humidity_mean': round(np.random.uniform(20, 98), 1),
                'relative_humidity_max': round(np.random.uniform(30, 100), 1),
                'dew_point_max': round(np.random.uniform(-5, 30), 1),
                'precipitation_sum': round(np.random.choice([0,0,0,0,1,2,5,10,15,20], p=[0.5,0.1,0.1,0.05,0.05,0.05,0.05,0.03,0.04,0.03]),1)
            })
    climate_df = pd.DataFrame(climate_records)

    air_records = []
    for city in cities:
        for i in range(num_days):
            date = (base_date + timedelta(days=i)).strftime('%Y-%m-%d')
            air_records.append({
                'city': city, 'date': date,
                'aqi': np.random.randint(10, 400), 'co': round(np.random.uniform(0.1, 15), 1),
                'no': round(np.random.uniform(0.5, 80), 1), 'no2': round(np.random.uniform(2, 120), 1),
                'o3': round(np.random.uniform(5, 250), 1), 'so2': round(np.random.uniform(0.5, 70), 1),
                'pm2_5': round(np.random.uniform(2, 300), 1), 'pm10': round(np.random.uniform(5, 350), 1),
                'nh3': round(np.random.uniform(0.2, 40), 1),
            })
    air_df = pd.DataFrame(air_records)

    soil_types = ['Alluvial', 'Black', 'Red', 'Laterite', 'Sandy', 'Loamy', 'Clay', 'Desert', 'Mountain', 'Peaty', 'Chalky', 'Silt']
    soil_records = []
    for city in cities:
        soil_records.append({
            'city': city, 'Soil Type': np.random.choice(soil_types),
            'Organic Matter (%)': round(np.random.uniform(0.2, 7.0), 1),
            'pH': round(np.random.uniform(4.0, 9.0), 1)
        })
    soil_df = pd.DataFrame(soil_records)

    water_sources = ['River', 'Lake', 'Reservoir', 'Groundwater', 'Canal', 'Desalination Plant', 'Spring', 'Coastal Aquifer', 'Glacier Melt']
    water_locations = ['Coastal', 'Inland Plains', 'Plateau', 'Delta Region', 'Hilly/Mountainous', 'Desert Oasis', 'River Valley']
    water_records = []
    for city in cities:
        min_dist = np.random.randint(0, 80)
        primary_dist = np.random.randint(min_dist, min_dist + 30) if min_dist < 70 else min_dist
        secondary_dist = np.random.randint(primary_dist, primary_dist + 40) if primary_dist < 90 else primary_dist
        water_records.append({
            'city': city, 'minimum distance': min_dist,
            'primary_source': np.random.choice(water_sources),
            'secondary_source': np.random.choice(water_sources),
            'location': np.random.choice(water_locations),
            'primary_source_distance': primary_dist,
            'secondary_source_distance': secondary_dist,
            'water_quality_index': np.random.randint(30, 98)
        })
    water_df = pd.DataFrame(water_records)

    return climate_df, air_df, soil_df, water_df

def convert_df_to_nested_dict(df, id_col='city', date_col=None, metrics_cols=None):
    nested_dict = {}
    for idx_val, group in df.groupby(id_col):
        city_data = {}
        if date_col:
            group[date_col] = group[date_col].astype(str)
            for _, row in group.iterrows():
                date_str = row[date_col]
                current_metrics_cols = metrics_cols if metrics_cols else [col for col in df.columns if col not in [id_col, date_col]]
                metrics = {col: row[col] for col in current_metrics_cols if col in row and pd.notna(row[col])}
                city_data[date_str] = metrics
        else:
            if not group.empty:
                metrics = group.iloc[0].to_dict()
                metrics.pop(id_col, None) # Remove the id_col from the metrics
                city_data = {k: v for k, v in metrics.items() if pd.notna(v)}
        nested_dict[idx_val] = city_data
    return nested_dict

@st.cache_data
def load_all_data():
    climate_df, air_df, soil_df, water_df = get_sample_data()
    climate_data_cols = [col for col in climate_df.columns if col not in ['city', 'date']]
    air_data_cols = [col for col in air_df.columns if col not in ['city', 'date']]
    loaded_data = {
        'climate_df': climate_df, # Keep DFs for easier export
        'air_df': air_df,
        'soil_df': soil_df,
        'water_df': water_df,
        'climate': convert_df_to_nested_dict(climate_df, id_col='city', date_col='date', metrics_cols=climate_data_cols),
        'air': convert_df_to_nested_dict(air_df, id_col='city', date_col='date', metrics_cols=air_data_cols),
        'soil': convert_df_to_nested_dict(soil_df, id_col='city'),
        'water': convert_df_to_nested_dict(water_df, id_col='city')
    }
    return loaded_data

data_dict = load_all_data()
climate_data = data_dict['climate']
air_data = data_dict['air']
soil_data = data_dict['soil']
water_data = data_dict['water']

# --- Gemini AI Model Initialization (Conditional) ---
GEMINI_MODEL = None
if GENAI_AVAILABLE:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.sidebar.warning(
                "Google API Key not found. AI features disabled. "
                "Set `GOOGLE_API_KEY` in `.streamlit/secrets.toml`.",
                icon="🔑"
            )
        else:
            genai.configure(api_key=api_key)
            GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
            st.sidebar.success("Google AI Model Initialized.", icon="🤖")
    except Exception as e:
        st.sidebar.error(f"Error initializing Google AI: {e}. AI features disabled.", icon="🔥")
else:
    st.sidebar.info("Google Generative AI library not installed. AI features disabled.", icon="ℹ️")


# --- Feature Functions (Health Score, Risk Assessment, Comparison) ---
def calculate_environmental_health_score(city_name):
    score_components = {'air_quality': 0, 'water_access_quality': 0, 'climate_stability': 0, 'soil_quality': 0}
    weights = {'air_quality': 0.35, 'water_access_quality': 0.25, 'climate_stability': 0.20, 'soil_quality': 0.20}

    if city_name in air_data and air_data[city_name]:
        aqi_values = [d.get('aqi', np.nan) for d in air_data[city_name].values() if isinstance(d, dict)]
        aqi_values = [v for v in aqi_values if pd.notna(v)]
        if aqi_values:
            avg_aqi = np.mean(aqi_values)
            if avg_aqi <= 50: score_components['air_quality'] = 100
            elif avg_aqi <= 100: score_components['air_quality'] = 75
            # ... (rest of AQI scoring)
            elif avg_aqi <= 300: score_components['air_quality'] = 10
            else: score_components['air_quality'] = 0

    if city_name in water_data and water_data[city_name]:
        wc_data = water_data[city_name]
        dist_score, qual_score = 0, 0
        distance = wc_data.get('minimum distance')
        if pd.notna(distance):
            if distance <= 1: dist_score = 100
            elif distance <= 25: dist_score = 50
            else: dist_score = 0
        quality = wc_data.get('water_quality_index')
        if pd.notna(quality): qual_score = max(0, min(100, quality)) # Assume WQI is 0-100
        score_components['water_access_quality'] = (0.6 * qual_score) + (0.4 * dist_score)

    if city_name in climate_data and climate_data[city_name]:
        temp_vars, temp_maxs, precip_sums = [], [], []
        for date_data in climate_data[city_name].values():
            if isinstance(date_data, dict):
                if pd.notna(date_data.get('temperature_max')) and pd.notna(date_data.get('temperature_min')):
                    temp_vars.append(abs(date_data['temperature_max'] - date_data['temperature_min']))
                    temp_maxs.append(date_data['temperature_max'])
                if pd.notna(date_data.get('precipitation_sum')): precip_sums.append(date_data['precipitation_sum'])
        
        temp_stab_score = max(0, 100 - np.mean(temp_vars) * 5) if temp_vars else 50
        extreme_heat_score = max(0, 100 - sum(1 for t in temp_maxs if t > 40) * 10) if temp_maxs else 50
        precip_score = 50
        if precip_sums:
            total_precip = np.sum(precip_sums)
            if 30 <= total_precip <= 300: precip_score = 80
            elif total_precip < 30 : precip_score = 30
            else: precip_score = 40 # Too wet
        score_components['climate_stability'] = (0.5 * temp_stab_score) + (0.3 * extreme_heat_score) + (0.2 * precip_score)


    if city_name in soil_data and soil_data[city_name]:
        sc_data = soil_data[city_name]
        om_score, ph_score = 0,0
        om = sc_data.get('Organic Matter (%)')
        if pd.notna(om):
            if om >= 4: om_score = 100
            elif om >=1: om_score = 50
            else: om_score = 25
        ph = sc_data.get('pH')
        if pd.notna(ph):
            if 6.0 <= ph <= 7.5: ph_score = 100
            elif (5.0 <= ph < 5.5) or (8.0 < ph <= 8.5): ph_score = 50
            else: ph_score = 25
        soil_type_scores = {'Alluvial': 95,'Black': 90,'Red': 75,'Laterite': 70,'Loamy': 85, 'Unknown': 30}
        type_base_score = soil_type_scores.get(sc_data.get('Soil Type', 'Unknown'), 30)
        score_components['soil_quality'] = (0.4 * type_base_score) + (0.3 * om_score) + (0.3 * ph_score)

    overall_score, total_weight = 0, 0
    for comp, score_val in score_components.items():
        if score_val > 0: overall_score += score_val * weights[comp]; total_weight += weights[comp]
    return (round(overall_score / total_weight, 1) if total_weight > 0 else 0), score_components

def assess_environmental_risks(city_name, business_type):
    risks = {'Air Pollution Risk': 'Low', 'Water Scarcity & Quality Risk': 'Low', 'Climate Change Impact Risk': 'Low', 'Soil Degradation Risk': 'Low'}
    risk_details = {}

    if city_name in air_data and air_data[city_name]:
        pm25_vals = [d.get('pm2_5', np.nan) for d in air_data[city_name].values() if isinstance(d, dict)]
        pm25_vals = [v for v in pm25_vals if pd.notna(v)]
        avg_pm25 = np.mean(pm25_vals) if pm25_vals else np.nan
        if pd.notna(avg_pm25):
            if avg_pm25 > 35: risks['Air Pollution Risk'] = 'High'
            elif avg_pm25 > 15: risks['Air Pollution Risk'] = 'Medium'
            risk_details['Air Pollution Risk'] = f"Avg PM2.5: {avg_pm25:.1f} µg/m³."
        else: risk_details['Air Pollution Risk'] = "Insufficient PM2.5 data."


    if city_name in water_data and water_data[city_name]:
        info = water_data[city_name]
        min_dist, wqi = info.get('minimum distance', np.nan), info.get('water_quality_index', np.nan)
        dist_risk, qual_risk = 0,0 # 0 low, 1 med, 2 high
        if pd.notna(min_dist):
            if min_dist > 40: dist_risk = 2
            elif min_dist > 20: dist_risk = 1
        if pd.notna(wqi):
            if wqi < 50: qual_risk = 2
            elif wqi < 70: qual_risk = 1
        
        combined_risk = max(dist_risk, qual_risk)
        if combined_risk == 2: risks['Water Scarcity & Quality Risk'] = 'High'
        elif combined_risk == 1: risks['Water Scarcity & Quality Risk'] = 'Medium'
        risk_details['Water Scarcity & Quality Risk'] = f"Nearest source: {min_dist if pd.notna(min_dist) else 'N/A'} km, WQI: {wqi if pd.notna(wqi) else 'N/A'}."


    if city_name in climate_data and climate_data[city_name]:
        max_temps = [d.get('temperature_max', np.nan) for d in climate_data[city_name].values() if isinstance(d, dict)]
        max_temps = [t for t in max_temps if pd.notna(t)]
        temp_risk = 0
        if max_temps:
            if np.mean(max_temps) > 38 or len([t for t in max_temps if t > 42]) > 3: temp_risk = 2
            elif np.mean(max_temps) > 35 : temp_risk = 1
        
        if temp_risk == 2: risks['Climate Change Impact Risk'] = 'High'
        elif temp_risk == 1: risks['Climate Change Impact Risk'] = 'Medium'
        risk_details['Climate Change Impact Risk'] = "Based on temperature extremes." # Simplified


    if city_name in soil_data and soil_data[city_name]:
        s_info = soil_data[city_name]
        om, ph = s_info.get('Organic Matter (%)', np.nan), s_info.get('pH', np.nan)
        om_risk, ph_risk = 0,0
        if pd.notna(om) and om < 1.0: om_risk = 2
        elif pd.notna(om) and om < 2.0: om_risk = 1
        if pd.notna(ph) and (ph < 5.0 or ph > 8.5): ph_risk = 2
        elif pd.notna(ph) and (ph < 5.5 or ph > 8.0): ph_risk = 1

        soil_combined_risk = max(om_risk, ph_risk)
        if soil_combined_risk == 2: risks['Soil Degradation Risk'] = 'High'
        elif soil_combined_risk == 1: risks['Soil Degradation Risk'] = 'Medium'
        risk_details['Soil Degradation Risk'] = f"OM: {om if pd.notna(om) else 'N/A'}%, pH: {ph if pd.notna(ph) else 'N/A'}."

    return risks, risk_details

def compare_cities(cities_list):
    comp_data = {}
    for city in cities_list:
        c_metrics = {}
        if city in air_data and air_data[city]:
            aqi_vals = [d.get('aqi',np.nan) for d in air_data[city].values() if isinstance(d,dict)]
            c_metrics['avg_aqi'] = round(np.nanmean([v for v in aqi_vals if pd.notna(v)]),1) if any(pd.notna(v) for v in aqi_vals) else np.nan
        if city in water_data and water_data[city]: c_metrics['water_quality_index'] = water_data[city].get('water_quality_index', np.nan)
        score, _ = calculate_environmental_health_score(city)
        c_metrics['health_score'] = score
        comp_data[city] = c_metrics
    return comp_data

# Original utility functions
def fetch_climate_data(city): return climate_data.get(city, {})
def filter_climate_by_variable(city_data_dict, variable):
    return {date: values[variable] for date, values in city_data_dict.items() if isinstance(values, dict) and variable in values and pd.notna(values[variable])}
def fetch_air_data(city): return air_data.get(city, {})
def filter_air_by_variable(city_data_dict, variable):
    return {date: values[variable] for date, values in city_data_dict.items() if isinstance(values, dict) and variable in values and pd.notna(values[variable])}

def visualize_soil_data(current_soil_data):
    soil_types_counts = pd.DataFrame([s_data.get("Soil Type", "Unknown") for s_data in current_soil_data.values()], columns=['Soil Type'])['Soil Type'].value_counts()
    if not soil_types_counts.empty:
        fig = px.pie(soil_types_counts, values=soil_types_counts.values, names=soil_types_counts.index, title="Soil Type Distribution", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    else: st.write("No soil data for pie chart.")

def fetch_water_data(city): return water_data.get(city, {})

# BST (preserved as per original, though its utility for AI prompts might be re-evaluated for complexity)
class TreeNode:
    def __init__(self, city_key): self.city_key = city_key; self.data_list = []; self.left = None; self.right = None
class BST:
    def __init__(self): self.root = None
    def insert(self, city_key, data_item): self.root = self._insert(self.root, city_key, data_item)
    def _insert(self, node, city_key, data_item):
        if node is None: node = TreeNode(city_key); node.data_list.append(data_item); return node
        if city_key < node.city_key: node.left = self._insert(node.left, city_key, data_item)
        elif city_key > node.city_key: node.right = self._insert(node.right, city_key, data_item)
        else: node.data_list.append(data_item)
        return node
    def search(self, city_key): return self._search(self.root, city_key)
    def _search(self, node, city_key):
        if node is None: return []
        if node.city_key == city_key: return node.data_list
        if city_key < node.city_key: return self._search(node.left, city_key)
        else: return self._search(node.right, city_key)

def search_city_in_bst(city_key_to_search, bst_instance):
    city_data_items = bst_instance.search(city_key_to_search)
    single_line_output = ""
    formatted_output_list = [] # For display
    if city_data_items:
        all_kv_pairs_for_ai = []
        for data_dict_item in city_data_items: # city_data_items is a list of dicts
            for key, value in data_dict_item.items(): # Iterate through each dict
                if isinstance(value, dict): # e.g. {"air_summary": {"avg_aqi": ...}}
                    for sub_key, sub_value in value.items():
                        val_str = f"{sub_value:.1f}" if isinstance(sub_value, float) else str(sub_value)
                        all_kv_pairs_for_ai.append(f"{key}_{sub_key}: {val_str}")
                else:
                    val_str = f"{value:.1f}" if isinstance(value, float) else str(value)
                    all_kv_pairs_for_ai.append(f"{key}: {val_str}")
            try: # For display purposes
                formatted_output_list.append(json.dumps(data_dict_item, indent=4, cls=NpEncoder))
            except TypeError:
                formatted_output_list.append(json.dumps({"error": "JSON serialization issue in BST display", "data": str(data_dict_item)}, indent=4))
        single_line_output = ', '.join(all_kv_pairs_for_ai) if all_kv_pairs_for_ai else f"No specific metrics processed for {city_key_to_search} for AI."
        formatted_output = "\n".join(formatted_output_list) if formatted_output_list else "No data to display."
    else:
        single_line_output = f"No data found for {city_key_to_search} in BST."
        formatted_output = single_line_output
    return single_line_output, formatted_output


# --- UI Layout Starts ---
all_available_cities = sorted(list(set(list(climate_data.keys()) + list(air_data.keys()) + list(soil_data.keys()) + list(water_data.keys()))))

with st.sidebar:
    st.header("🧭 Navigation")
    page = st.radio("Go to", ("🏠 Home", "📊 Analysis", "📈 Visualization", "🆚 City Comparison", "⚡ Quick Assessment", "📋 Reports"))
    st.markdown("---")
    if all_available_cities:
        st.header("🌍 Quick City Info")
        quick_city_sb = st.selectbox("Select a city:", [""] + all_available_cities, key="sb_quick_city", help="Get a quick environmental snapshot.")
        if quick_city_sb:
            score_sb, _ = calculate_environmental_health_score(quick_city_sb)
            st.metric("Env. Health Score", f"{score_sb:.1f}/100")
            risks_sb, _ = assess_environmental_risks(quick_city_sb, "General")
            high_risks_count_sb = sum(1 for r_val in risks_sb.values() if r_val == 'High')
            if high_risks_count_sb > 0: st.error(f"⚠️ {high_risks_count_sb} High Risk Area(s)")
            else: st.success("✅ Low Overall Risk Profile")

# HOME PAGE
if page == "🏠 Home":
    col1_home, col2_home = st.columns([2, 1])
    with col1_home:
        st.markdown("""
        ## Welcome to the Advanced Environmental Impact Analyzer! 🌍
        Our platform empowers informed decisions by analyzing air quality, climate patterns, soil characteristics, and water resource availability. 
        Leverage AI-Powered Recommendations (if API key is configured), comprehensive Risk Assessments, insightful City Comparisons, and data-driven Environmental Health Scoring.
        """)
        try:
            st.image("assets/Simple_Environment.jpeg", use_container_width=True, caption="Sustainable Futures Start Here")
        except FileNotFoundError:
            st.markdown("<div class='alert-box alert-warning'>ℹ️ Banner image not found. Create an 'assets' folder with '5825745.jpg' or update path.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<div class='alert-box alert-danger'>🔥 Error loading banner: {e}</div>", unsafe_allow_html=True)

    with col2_home:
        st.markdown("### 📊 **Platform Stats (Sample Data)**")
        st.markdown(f"<div class='metric-card'>Cities with Climate Data: <strong>{len(climate_data)}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Cities with Air Data: <strong>{len(air_data)}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Cities with Soil Data: <strong>{len(soil_data)}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Cities with Water Data: <strong>{len(water_data)}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'>Total Unique Cities: <strong>{len(all_available_cities)}</strong></div>", unsafe_allow_html=True)
    
    st.markdown("---<br>", unsafe_allow_html=True)
    st.subheader("🌟 Featured Capabilities")
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.markdown("<div class='metric-card'><h4><center>🏭 Business Impact</center></h4><p>Tailored assessments & AI recommendations (if enabled).</p></div>", unsafe_allow_html=True)
    fcol2.markdown("<div class='metric-card'><h4><center>⚡ Real-time Risk</center></h4><p>Instant evaluation of key environmental risks.</p></div>", unsafe_allow_html=True)
    fcol3.markdown("<div class='metric-card'><h4><center>📈 Interactive Visuals</center></h4><p>Dynamic charts for trends and comparative analysis.</p></div>", unsafe_allow_html=True)

# ANALYSIS PAGE
elif page == "📊 Analysis":
    st.header("🔬 Comprehensive Environmental Analysis")
    tab_an1, tab_an2, tab_an3 = st.tabs(["🏭 Business Analysis", "💚 Health Score", "⚠️ Risk Assessment"])

    with tab_an1:
        st.subheader("Business Environmental Impact Analysis")
        col_an1, col_an2 = st.columns(2)
        city_name_an = col_an1.selectbox("🏙️ Select City:", [""] + all_available_cities, key="an_city", help="Choose a city for analysis.")
        business_an = col_an1.text_input("🏢 Enter Business/Activity Type:", key="an_biz", placeholder="e.g., Textile Manufacturing, Urban Farming")
        
        # Dates are for context display, not strict filtering here
        default_start_date = datetime.now() - timedelta(days=30)
        default_end_date = datetime.now()
        start_d_an_str = col_an2.date_input("📅 Analysis Start Date (Context):", default_start_date, key="an_start").strftime('%Y-%m-%d')
        end_d_an_str = col_an2.date_input("📅 Analysis End Date (Context):", default_end_date, key="an_end").strftime('%Y-%m-%d')

        if st.button("🔍 Generate Analysis", type="primary", key="an_fetch_btn"):
            if city_name_an and business_an:
                with st.spinner("Analyzing environmental data and generating insights..."):
                    bst_an = BST()
                    summary_for_bst = []
                    # Simplified summary for BST for AI prompt
                    if city_name_an in air_data and air_data[city_name_an]:
                        aqi_vals_an = [d.get('aqi', np.nan) for d in air_data[city_name_an].values() if isinstance(d, dict)]
                        avg_aqi_an = np.nanmean([v for v in aqi_vals_an if pd.notna(v)]) if any(pd.notna(v) for v in aqi_vals_an) else 'N/A'
                        summary_for_bst.append({"air_summary": {"avg_aqi": avg_aqi_an}})
                    if city_name_an in climate_data and climate_data[city_name_an]:
                        temp_vals_an = [d.get('temperature_mean', np.nan) for d in climate_data[city_name_an].values() if isinstance(d, dict)]
                        avg_temp_an = np.nanmean([v for v in temp_vals_an if pd.notna(v)]) if any(pd.notna(v) for v in temp_vals_an) else 'N/A'
                        summary_for_bst.append({"climate_summary": {"avg_temp_mean": avg_temp_an}})
                    if city_name_an in soil_data and soil_data[city_name_an]: summary_for_bst.append({"soil_info": soil_data[city_name_an]})
                    if city_name_an in water_data and water_data[city_name_an]: summary_for_bst.append({"water_info": water_data[city_name_an]})

                    if summary_for_bst:
                        for item in summary_for_bst: bst_an.insert(city_name_an, item)
                    else:
                        bst_an.insert(city_name_an, {"error": "No summary data available for BST."})

                    single_line_ai, formatted_display_ai = search_city_in_bst(city_name_an, bst_an)

                    res_col1, res_col2 = st.columns([2,1])
                    with res_col1:
                        st.subheader(f"📋 Analysis for: {business_an}")
                        st.write(f"**Location:** {city_name_an}")
                        st.write(f"**Contextual Period:** {start_d_an_str} to {end_d_an_str}")
                        with st.expander("📄 View Summarized Data Used for AI (JSON)", expanded=False):
                            st.code(formatted_display_ai, language='json')
                    with res_col2:
                        score_an_val, _ = calculate_environmental_health_score(city_name_an)
                        st.metric("🌿 Env. Health Score", f"{score_an_val:.1f}/100")
                        risks_an_val, _ = assess_environmental_risks(city_name_an, business_an)
                        high_risks_an_val_count = sum(1 for r in risks_an_val.values() if r == 'High')
                        if high_risks_an_val_count > 0: st.error(f"⚠️ {high_risks_an_val_count} High Risk Factor(s)")
                        else: st.success("✅ Low Overall Risk Profile")

                    if GEMINI_MODEL and single_line_ai and "No data found" not in single_line_ai and "No summary data available" not in single_line_ai and "No specific metrics processed" not in single_line_ai:
                        try:
                            prompt = f"""
                            You are an environmental consultant. Provide a concise environmental impact analysis for a '{business_an}' looking to operate in '{city_name_an}'.
                            Focus on the period between {start_d_an_str} and {end_d_an_str}.
                            Key environmental data summary for {city_name_an}: {single_line_ai}.

                            Structure your response as follows:
                            **1. Key Environmental Considerations (3-4 bullet points):** Based on the provided data, what are the most salient environmental factors for this business type in this city?
                            **2. Potential Negative Impacts from '{business_an}' (2-3 bullet points):** What are common negative environmental impacts this type of business might cause, considering the city's profile?
                            **3. Suggested Mitigation Strategies (3-4 bullet points):** Propose actionable mitigation strategies relevant to the business and the city's environmental data.
                            **4. Overall Suitability & Outlook (brief paragraph):** A short assessment of the city's environmental suitability for this business, highlighting opportunities or critical challenges.
                            """
                            response = GEMINI_MODEL.generate_content(prompt)
                            st.markdown("---")
                            st.subheader("🤖 AI-Generated Insights & Recommendations")
                            st.markdown(response.text)
                        except Exception as e_ai:
                            st.markdown(f"<div class='alert-box alert-danger'>🔥 Error generating AI recommendations: {e_ai}</div>", unsafe_allow_html=True)
                    elif not GEMINI_MODEL:
                        st.markdown("<div class='alert-box alert-warning'>🤖 AI-powered insights are unavailable. Google AI Model not initialized (check API key in secrets).</div>", unsafe_allow_html=True)
                    else: # Cases where AI can't run due to insufficient data
                         st.markdown(f"<div class='alert-box alert-warning'>🤖 AI insights cannot be generated for {city_name_an} due to insufficient processed data. Basic analysis is still available.</div>", unsafe_allow_html=True)
            else:
                st.error("⚠️ Please select a city and enter a business/activity type to proceed.", icon="❗")

    with tab_an2:
        st.subheader("💚 Environmental Health Score Analysis")
        city_for_health_sel = st.selectbox("Select city for detailed health analysis:", [""] + all_available_cities, key="health_city_select_tab")
        if city_for_health_sel:
            score_hs, components_hs = calculate_environmental_health_score(city_for_health_sel)
            col_hs1, col_hs2 = st.columns([1, 2])
            with col_hs1:
                st.metric("🌿 Overall Health Score", f"{score_hs:.1f}/100")
                if score_hs == 0 and not any(components_hs.values()): # Check if score is 0 due to no data
                     st.warning("No data available to calculate health score.")
                elif score_hs >= 75: st.markdown("<div class='alert-box alert-success'>🟢 Excellent environmental conditions.</div>", unsafe_allow_html=True)
                elif score_hs >= 60: st.markdown("<div class='alert-box alert-success' style='background-color: #e6ffe6;'> Borderline Good/Excellent conditions.</div>", unsafe_allow_html=True) # Custom for borderline
                elif score_hs >= 40: st.markdown("<div class='alert-box alert-warning'>🟠 Moderate environmental conditions. Areas for improvement exist.</div>", unsafe_allow_html=True)
                else: st.markdown("<div class='alert-box alert-danger'>🔴 Poor environmental conditions. Significant concerns.</div>", unsafe_allow_html=True)
            with col_hs2:
                st.subheader("📊 Score Components Breakdown")
                # Filter for components that contributed (score > 0) or are default categories
                valid_components = {
                    k.replace('_',' ').title(): v for k,v in components_hs.items()
                    if v > 0 or k in ['air_quality', 'water_access_quality', 'climate_stability', 'soil_quality']
                }
                if valid_components and any(v > 0 for v in valid_components.values()): # Ensure there's something to plot
                    component_df_hs = pd.DataFrame(list(valid_components.items()), columns=['Component', 'Score'])
                    fig_hs = px.bar(component_df_hs, x='Component', y='Score',
                                    title=f"Health Score Components for {city_for_health_sel}",
                                    color='Score', color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[0,100],
                                    labels={'Score':'Component Score (0-100)'})
                    fig_hs.update_layout(xaxis_title="Environmental Aspect", yaxis_title="Calculated Score")
                    st.plotly_chart(fig_hs, use_container_width=True)
                else:
                    st.info(f"Not enough component data to display breakdown for {city_for_health_sel}.")

    with tab_an3:
        st.subheader("⚠️ Environmental Risk Assessment")
        col_ra1, col_ra2 = st.columns(2)
        risk_city_sel = col_ra1.selectbox("Select city for risk assessment:", [""] + all_available_cities, key="risk_city_select_tab")
        risk_business_inp = col_ra2.text_input("Business type (optional context):", placeholder="e.g., Agriculture, Tourism", key="risk_business_input_tab")
        
        if st.button("🛡️ Assess Risks", key="assess_risk_btn_tab"):
            if risk_city_sel:
                risks_ra, risk_details_ra = assess_environmental_risks(risk_city_sel, risk_business_inp or "General")
                st.subheader(f"🎯 Risk Profile for {risk_city_sel}")
                if risk_business_inp: st.caption(f"Context: {risk_business_inp}")

                risk_counts_ra = {'High': 0, 'Medium': 0, 'Low': 0, 'Unknown':0}
                for r_level in risks_ra.values(): risk_counts_ra[r_level] +=1
                
                rscol1,rscol2,rscol3 = st.columns(3)
                rscol1.metric("🔴 High Risk Factors", risk_counts_ra['High'])
                rscol2.metric("🟡 Medium Risk Factors", risk_counts_ra['Medium'])
                rscol3.metric("🟢 Low Risk Factors", risk_counts_ra['Low'])

                st.markdown("---")
                st.subheader("📋 Detailed Risk Analysis & Insights")
                for risk_type_ra, risk_level_ra in risks_ra.items():
                    alert_class_ra, icon_ra = "", ""
                    if risk_level_ra == 'High': alert_class_ra, icon_ra = "alert-danger", "🔴"
                    elif risk_level_ra == 'Medium': alert_class_ra, icon_ra = "alert-warning", "🟡"
                    else: alert_class_ra, icon_ra = "alert-success", "🟢"
                    
                    detail_ra = risk_details_ra.get(risk_type_ra, "No specific details available for this risk category.")
                    st.markdown(f"""
                    <div class="alert-box {alert_class_ra}">
                        <strong>{icon_ra} {risk_type_ra}: {risk_level_ra}</strong><br>
                        <small>{detail_ra}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Please select a city to assess risks.", icon="❗")

# VISUALIZATION PAGE
elif page == "📈 Visualization":
    st.header("📊 Interactive Environmental Data Visualization")
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["🌦️ Climate Trends", "💨 Air Quality", "🌱 Soil Analysis", "💧 Water Resources"])

    with viz_tab1: # Climate
        st.subheader("🌦️ Climate Data Trends")
        if not climate_data:
            st.warning("No climate data loaded to visualize.")
        else:
            col_c1_viz, col_c2_viz = st.columns(2)
            sel_clim_city_viz = col_c1_viz.selectbox("Select City:", [""] + sorted(list(climate_data.keys())), key="viz_clim_city_sel")
            clim_vars_viz = ["temperature_mean", "temperature_max", "precipitation_sum", "relative_humidity_mean", "wind_speed_mean"]
            sel_clim_var_viz = col_c2_viz.selectbox("Select Variable:", clim_vars_viz, key="viz_clim_var_sel")

            if sel_clim_city_viz and sel_clim_var_viz:
                clim_city_data_dict = fetch_climate_data(sel_clim_city_viz)
                filtered_clim_dict = filter_climate_by_variable(clim_city_data_dict, sel_clim_var_viz)
                if filtered_clim_dict:
                    clim_df_viz = pd.DataFrame.from_dict(filtered_clim_dict, orient='index', columns=[sel_clim_var_viz])
                    try:
                        clim_df_viz.index = pd.to_datetime(clim_df_viz.index)
                    except ValueError:
                        st.warning("Could not parse dates for climate data, plotting with string index. Trend may not be accurate.")
                    clim_df_viz = clim_df_viz.sort_index()
                    fig_clim_viz = px.line(clim_df_viz, y=sel_clim_var_viz, title=f"{sel_clim_var_viz.replace('_',' ').title()} Trend in {sel_clim_city_viz}", markers=True)
                    fig_clim_viz.update_layout(xaxis_title="Date", yaxis_title=sel_clim_var_viz.replace('_',' ').title())
                    st.plotly_chart(fig_clim_viz, use_container_width=True)
                    with st.expander("View Data Summary"):
                        st.dataframe(clim_df_viz.describe())
                else:
                    st.info(f"No data available for '{sel_clim_var_viz}' in {sel_clim_city_viz}.")
            elif sel_clim_city_viz:
                 st.info("Please select a climate variable to visualize.")


    with viz_tab2: # Air Quality
        st.subheader("💨 Air Quality Visualization")
        if not air_data:
            st.warning("No air quality data loaded to visualize.")
        else:
            col_a1_viz, col_a2_viz = st.columns(2)
            sel_air_city_viz = col_a1_viz.selectbox("Select City:", [""] + sorted(list(air_data.keys())), key="viz_air_city_sel")
            air_vars_viz = ["aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3"]
            sel_air_var_viz = col_a2_viz.selectbox("Select Pollutant:", air_vars_viz, key="viz_air_var_sel")

            if sel_air_city_viz and sel_air_var_viz:
                air_city_data_dict_viz = fetch_air_data(sel_air_city_viz)
                filtered_air_dict_viz = filter_air_by_variable(air_city_data_dict_viz, sel_air_var_viz)
                if filtered_air_dict_viz:
                    air_df_viz_tab = pd.DataFrame.from_dict(filtered_air_dict_viz, orient='index', columns=[sel_air_var_viz])
                    try:
                        air_df_viz_tab.index = pd.to_datetime(air_df_viz_tab.index)
                    except ValueError:
                        st.warning("Could not parse dates for air quality data, plotting with string index. Trend may not be accurate.")
                    air_df_viz_tab = air_df_viz_tab.sort_index()
                    fig_air_viz = px.line(air_df_viz_tab, y=sel_air_var_viz, title=f"{sel_air_var_viz.upper()} Levels Trend in {sel_air_city_viz}", markers=True)
                    fig_air_viz.update_layout(xaxis_title="Date", yaxis_title=sel_air_var_viz.upper())
                    st.plotly_chart(fig_air_viz, use_container_width=True)
                    with st.expander("View Pollutant Data Summary"):
                        st.dataframe(air_df_viz_tab.describe())
                else:
                    st.info(f"No data available for '{sel_air_var_viz}' in {sel_air_city_viz}.")
            elif sel_air_city_viz:
                 st.info("Please select an air quality pollutant to visualize.")

    with viz_tab3: # Soil Analysis
        st.subheader("🌱 Soil Analysis Visualization")
        if not soil_data:
            st.warning("No soil data loaded to visualize.")
        else:
            visualize_soil_data(soil_data) # Pie chart for overall distribution
            st.markdown("---")
            st.subheader("Compare Soil Metrics Across Cities")
            soil_df_for_viz = pd.DataFrame([{'City': city, **s_data} for city, s_data in soil_data.items() if s_data]) # ensure s_data is not empty
            
            if not soil_df_for_viz.empty:
                soil_metric_sel_viz = st.selectbox("Select Soil Metric to Compare:", ['pH', 'Organic Matter (%)'], key="viz_soil_metric")
                if soil_metric_sel_viz in soil_df_for_viz.columns:
                    # Ensure the metric column is numeric for sorting
                    soil_df_for_viz[soil_metric_sel_viz] = pd.to_numeric(soil_df_for_viz[soil_metric_sel_viz], errors='coerce')
                    soil_df_for_viz_sorted = soil_df_for_viz.dropna(subset=[soil_metric_sel_viz]).sort_values(soil_metric_sel_viz, ascending=False).head(25)

                    if not soil_df_for_viz_sorted.empty:
                        fig_soil_bar_viz = px.bar(soil_df_for_viz_sorted, x='City', y=soil_metric_sel_viz,
                                                color='Soil Type', hover_data=['pH', 'Organic Matter (%)'],
                                                title=f"Top Cities by Soil {soil_metric_sel_viz.replace('_',' ').title()}")
                        st.plotly_chart(fig_soil_bar_viz, use_container_width=True)
                    else:
                        st.info(f"No valid data to display for soil metric '{soil_metric_sel_viz}' after filtering.")
                else:
                    st.info("Selected soil metric not found in the data.")
            else:
                st.info("Not enough soil data for comparative visualization.")


    with viz_tab4: # Water Resources
        st.subheader("💧 Water Resources Analysis")
        if not water_data:
            st.warning("No water data loaded to visualize.")
        else:
            sel_water_city_viz_tab = st.selectbox("Select City for Detailed Water View:", [""]+sorted(list(water_data.keys())), key="viz_water_city_tab_sel")
            if sel_water_city_viz_tab:
                water_city_info_viz_tab = fetch_water_data(sel_water_city_viz_tab)
                if water_city_info_viz_tab:
                    wqi_gauge_val = water_city_info_viz_tab.get('water_quality_index')
                    if pd.notna(wqi_gauge_val):
                        fig_wqi_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=wqi_gauge_val,
                            title={'text': f"Water Quality Index (WQI) - {sel_water_city_viz_tab}"},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"},
                                   'steps': [{'range': [0, 50], 'color': "rgba(211, 77, 86, 0.7)"}, # Light Red
                                             {'range': [50, 80], 'color': "rgba(255, 216, 107, 0.7)"}, # Light Yellow
                                             {'range': [80, 100], 'color': "rgba(111, 185, 119, 0.7)"}]})) # Light Green
                        st.plotly_chart(fig_wqi_gauge, use_container_width=True)
                    else:
                        st.info(f"Water Quality Index not available for {sel_water_city_viz_tab}.")
                    
                    st.write("Other Water Metrics:")
                    display_water_metrics = {k.replace('_',' ').title(): v for k,v in water_city_info_viz_tab.items() if k != 'water_quality_index'}
                    if display_water_metrics:
                        st.table(pd.Series(display_water_metrics, name="Value"))
                    else:
                        st.caption("No other water metrics to display.")
                else:
                    st.info(f"No water data found for {sel_water_city_viz_tab}.")
            st.markdown("---")
            st.subheader("Compare Water Metrics Across Cities")
            water_df_for_viz = pd.DataFrame([{'City': city, **w_data} for city, w_data in water_data.items() if w_data])
            if not water_df_for_viz.empty:
                water_metric_sel_viz = st.selectbox("Select Water Metric to Compare:",
                                                    ['minimum distance', 'water_quality_index', 'primary_source_distance', 'secondary_source_distance'],
                                                    key="viz_water_metric_comp")
                if water_metric_sel_viz in water_df_for_viz.columns:
                    water_df_for_viz[water_metric_sel_viz] = pd.to_numeric(water_df_for_viz[water_metric_sel_viz], errors='coerce')
                    # Sort ascending for distance, descending for quality index
                    ascending_sort = 'distance' in water_metric_sel_viz
                    water_df_viz_sorted = water_df_for_viz.dropna(subset=[water_metric_sel_viz]).sort_values(water_metric_sel_viz, ascending=ascending_sort).head(25)

                    if not water_df_viz_sorted.empty:
                        fig_water_comp_viz = px.bar(water_df_viz_sorted, x='City', y=water_metric_sel_viz,
                                                    color='primary_source', hover_data=water_df_for_viz.columns.tolist(),
                                                    title=f"{('Top' if not ascending_sort else 'Cities with Shortest')} {water_metric_sel_viz.replace('_',' ').title()}")
                        st.plotly_chart(fig_water_comp_viz, use_container_width=True)
                    else:
                        st.info(f"No valid data for '{water_metric_sel_viz}' after filtering.")
            else:
                st.info("Not enough water data for comparative visualization.")


# CITY COMPARISON PAGE
elif page == "🆚 City Comparison":
    st.header("🏙️ Comparative City Analysis")
    if len(all_available_cities) >= 2:
        default_comp_cities = all_available_cities[:min(len(all_available_cities), 3)] # Default to 3 cities
        sel_cities_comp_tab = st.multiselect("Select cities to compare (2-10 recommended):", all_available_cities,
                                         default=default_comp_cities, key="comp_city_multi_tab", max_selections=10,
                                         help="Choose multiple cities to see a side-by-side comparison of key metrics.")
        if len(sel_cities_comp_tab) >= 2:
            comparison_data_dict_tab = compare_cities(sel_cities_comp_tab)
            comparison_df_comp_tab = pd.DataFrame.from_dict(comparison_data_dict_tab, orient='index').reset_index().rename(columns={'index':'City'})
            
            if not comparison_df_comp_tab.empty:
                st.subheader("🏥 Environmental Health Score Comparison")
                if 'health_score' in comparison_df_comp_tab.columns and comparison_df_comp_tab['health_score'].notna().any():
                    fig_comp_hs_tab = px.bar(comparison_df_comp_tab.sort_values('health_score', ascending=False),
                                       x='City', y='health_score', title="Health Scores by City",
                                       color='health_score', color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[0,100],
                                       labels={'health_score':'Overall Health Score (0-100)'})
                    st.plotly_chart(fig_comp_hs_tab, use_container_width=True)
                else:
                    st.info("Health score data not available or insufficient for selected cities for this chart.")

                metrics_to_plot_comp = ['avg_aqi', 'water_quality_index'] # Add more as needed from compare_cities()
                available_metrics_comp = [m for m in metrics_to_plot_comp if m in comparison_df_comp_tab.columns and comparison_df_comp_tab[m].notna().any()]
                
                if available_metrics_comp:
                    st.subheader("📊 Key Metric Comparison (Average AQI, Water Quality Index)")
                    df_melt_comp = comparison_df_comp_tab.melt(id_vars=['City'], value_vars=available_metrics_comp,
                                                               var_name='Metric', value_name='Value')
                    df_melt_comp.dropna(subset=['Value'], inplace=True) # Ensure no NaNs are plotted in grouped bar

                    if not df_melt_comp.empty:
                        fig_grouped_bar_comp = px.bar(df_melt_comp, x='City', y='Value', color='Metric', barmode='group',
                                                    title="Key Environmental Metrics Comparison")
                        st.plotly_chart(fig_grouped_bar_comp, use_container_width=True)
                    else:
                        st.info("No data available for selected key metrics (Avg AQI, WQI) for the chosen cities.")
                else:
                    st.info("No comparable key metrics (Avg AQI, WQI) found for the selected cities.")

                st.subheader("📈 Detailed Comparison Table")
                st.dataframe(comparison_df_comp_tab.set_index('City').round(1).fillna("N/A"), use_container_width=True) # Round and fill NaN for display
            else:
                st.warning("Could not generate comparison data for the selected cities.")
        elif sel_cities_comp_tab: # If some cities selected but < 2
             st.info("Please select at least 2 cities for comparison.")
    else:
        st.warning("Not enough city data (at least 2 cities required) loaded for comparison features.")


# QUICK ASSESSMENT PAGE
elif page == "⚡ Quick Assessment":
    st.header("⚡ Quick Environmental Assessment")
    quick_assess_city_sel_tab = st.selectbox("🌍 Select city for a rapid assessment:", [""] + all_available_cities, key="qa_city_select_tab")
    if quick_assess_city_sel_tab:
        with st.spinner(f"Generating quick assessment for {quick_assess_city_sel_tab}..."):
            score_qa, components_qa = calculate_environmental_health_score(quick_assess_city_sel_tab)
            risks_qa, risk_details_qa = assess_environmental_risks(quick_assess_city_sel_tab, "General")
            
            st.subheader(f"📋 Quick Environmental Snapshot: {quick_assess_city_sel_tab}")
            
            col_qa1, col_qa2 = st.columns(2)
            with col_qa1:
                st.metric("🌿 Env. Health Score", f"{score_qa:.1f}/100")
                if score_qa >= 75: st.success("Overall: Good to Excellent")
                elif score_qa >= 40: st.warning("Overall: Moderate")
                else: st.error("Overall: Poor to Concerning")

            with col_qa2:
                high_risks_qa_count = sum(1 for r_val in risks_qa.values() if r_val == 'High')
                st.metric("🔴 High Risk Factors", high_risks_qa_count)
                if high_risks_qa_count > 0: st.error(f"{high_risks_qa_count} critical risk(s) identified.")
                else: st.success("No high environmental risks detected.")

            st.markdown("---")
            st.subheader("📊 Health Score Components")
            valid_components_qa = {
                k.replace('_',' ').title(): v for k,v in components_qa.items()
                if v > 0 or k in ['air_quality', 'water_access_quality', 'climate_stability', 'soil_quality']
            }
            if valid_components_qa and any(v > 0 for v in valid_components_qa.values()):
                 fig_qa_pie = px.pie(values=list(valid_components_qa.values()), names=list(valid_components_qa.keys()),
                                     title="Contribution to Environmental Health Score", hole=0.4,
                                     color_discrete_sequence=px.colors.sequential.Greens_r) # Example color sequence
                 fig_qa_pie.update_traces(textposition='inside', textinfo='percent+label')
                 st.plotly_chart(fig_qa_pie, use_container_width=True)
            else:
                 st.info(f"Not enough component data for health score breakdown for {quick_assess_city_sel_tab}.")
            
            st.markdown("---")
            st.subheader("⚠️ Key Risk Areas")
            for risk_type_qa, risk_level_qa in risks_qa.items():
                if risk_level_qa != 'Low': # Highlight Medium and High risks
                    alert_class_qa, icon_qa = ("alert-danger", "🔴") if risk_level_qa == 'High' else ("alert-warning", "🟡")
                    detail_qa = risk_details_qa.get(risk_type_qa, "Details not available.")
                    st.markdown(f"""<div class="alert-box {alert_class_qa}"><strong>{icon_qa} {risk_type_qa}: {risk_level_qa}</strong><br><small>{detail_qa}</small></div>""", unsafe_allow_html=True)
            if all(r == 'Low' for r in risks_qa.values()):
                st.markdown("<div class='alert-box alert-success'>✅ All assessed risk categories are currently Low.</div>", unsafe_allow_html=True)


# REPORTS PAGE
elif page == "📋 Reports":
    st.header("📋 Environmental Reports & Analytics")
    tab_rep1, tab_rep2, tab_rep3 = st.tabs(["📊 Summary Report", "📈 Trend Analysis", "📥 Data Export"])

    with tab_rep1:
        st.subheader("📊 Overall Environmental Summary (Sample Data)")
        if all_available_cities:
            st.markdown(f"This report summarizes data across **{len(all_available_cities)}** unique sampled cities.")
            
            st.subheader("🏥 Environmental Health Score Distribution")
            health_scores_rep_tab = [calculate_environmental_health_score(c)[0] for c in all_available_cities if c]
            health_scores_rep_tab = [s for s in health_scores_rep_tab if pd.notna(s) and s > 0] # Filter out NaNs and 0s if they mean no data

            if health_scores_rep_tab:
                fig_rep_hist = px.histogram(pd.DataFrame(health_scores_rep_tab, columns=['Health Score']),
                                            x='Health Score', nbins=10,
                                            title='Distribution of Environmental Health Scores Across All Cities',
                                            labels={'Health Score':'Health Score Range'},
                                            color_discrete_sequence=['#2E8B57'])
                fig_rep_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_rep_hist, use_container_width=True)
                avg_health_score = np.mean(health_scores_rep_tab)
                st.markdown(f"**Average Health Score (across cities with data):** {avg_health_score:.1f}/100")
            else:
                st.info("Not enough health score data available across cities for a distribution plot.")
            
            # Add more summary stats if needed, e.g., average AQI across all cities
        else:
            st.warning("No city data loaded to generate a summary report.")

    with tab_rep2: # Trend Analysis
        st.subheader("📈 Parameter Trend Analysis Over Time")
        if not (climate_data or air_data):
            st.warning("No time-series data (Climate or Air Quality) loaded for trend analysis.")
        else:
            trend_city_sel = st.selectbox("Select City for Trend Analysis:", [""] + all_available_cities, key="trend_city_sel_rep")
            if trend_city_sel:
                data_types_with_time = []
                if trend_city_sel in climate_data and climate_data[trend_city_sel]: data_types_with_time.append("Climate")
                if trend_city_sel in air_data and air_data[trend_city_sel]: data_types_with_time.append("Air Quality")

                if not data_types_with_time:
                    st.info(f"No time-series data (Climate/Air) found for {trend_city_sel}.")
                else:
                    trend_data_type_sel = st.selectbox("Select Data Type:", data_types_with_time, key="trend_data_type_sel_rep")
                    
                    available_metrics_trend = []
                    data_for_trend = None

                    if trend_data_type_sel == "Climate":
                        data_for_trend = climate_data.get(trend_city_sel, {})
                        if data_for_trend and isinstance(next(iter(data_for_trend.values()), None), dict):
                            available_metrics_trend = sorted(list(next(iter(data_for_trend.values())).keys()))
                    elif trend_data_type_sel == "Air Quality":
                        data_for_trend = air_data.get(trend_city_sel, {})
                        if data_for_trend and isinstance(next(iter(data_for_trend.values()), None), dict):
                            available_metrics_trend = sorted(list(next(iter(data_for_trend.values())).keys()))

                    if available_metrics_trend:
                        trend_metric_sel = st.selectbox("Select Metric:", available_metrics_trend, key="trend_metric_sel_rep")
                        
                        if st.button("📊 Generate Trend Plot", key="trend_plot_btn_rep"):
                            if data_for_trend and trend_metric_sel:
                                metric_values = {}
                                for date_str, daily_data in data_for_trend.items():
                                    if isinstance(daily_data, dict) and trend_metric_sel in daily_data and pd.notna(daily_data[trend_metric_sel]):
                                        try:
                                            parsed_date = pd.to_datetime(date_str)
                                            metric_values[parsed_date] = daily_data[trend_metric_sel]
                                        except (ValueError, TypeError):
                                            continue # Skip unparseable dates quietly for now

                                if metric_values:
                                    trend_df = pd.DataFrame(list(metric_values.items()), columns=['Date', 'Value']).set_index('Date').sort_index()
                                    fig_trend = px.line(trend_df, y='Value', title=f"{trend_metric_sel.replace('_',' ').title()} Trend for {trend_city_sel}", markers=True)
                                    fig_trend.update_xaxes(title_text='Date')
                                    fig_trend.update_yaxes(title_text=trend_metric_sel.replace('_',' ').title())
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                else:
                                    st.warning(f"No valid data points found for '{trend_metric_sel}' in {trend_city_sel} for the trend plot.")
                            else:
                                st.error("Please ensure city, data type, and metric are selected, and data is available.")
                    elif trend_city_sel and trend_data_type_sel:
                        st.warning(f"No plottable metrics available for {trend_data_type_sel} in {trend_city_sel}. Data might be missing or in an unexpected format.")
            else:
                st.info("Select a city to begin trend analysis.")

    with tab_rep3: # Data Export
        st.subheader("📥 Data Export Options")
        export_city_sel_tab = st.selectbox("Select city for data export:", ["All Sample Data"] + all_available_cities, key="export_city_rep_tab")
        export_format_sel_tab = st.radio("Select export format:", ["JSON", "CSV"], key="export_format_rep_tab", horizontal=True)

        if st.button("📄 Generate & Download File(s)", key="export_gen_btn_tab"):
            with st.spinner("Preparing data for export..."):
                filename_base = "environmental_data"
                
                if export_city_sel_tab == "All Sample Data":
                    filename_base = "all_cities_environmental"
                    if export_format_sel_tab == "JSON":
                        all_data_to_export = {
                            'climate': climate_data, 'air': air_data,
                            'soil': soil_data, 'water': water_data
                        }
                        try:
                            json_data_all = json.dumps(all_data_to_export, indent=4, cls=NpEncoder)
                            st.download_button(label="📥 Download All Data (JSON)", data=json_data_all,
                                            file_name=f"{filename_base}.json", mime="application/json", key="download_all_json")
                        except Exception as e:
                            st.error(f"Error generating JSON for all data: {e}")

                    elif export_format_sel_tab == "CSV":
                        st.info("For 'All Sample Data' in CSV format, individual files for each data type will be provided below.")
                        data_frames_all = {
                            "climate": data_dict['climate_df'], "air": data_dict['air_df'],
                            "soil": data_dict['soil_df'], "water": data_dict['water_df']
                        }
                        for df_name, df_content in data_frames_all.items():
                            if not df_content.empty:
                                csv_data = df_content.to_csv(index=False).encode('utf-8')
                                st.download_button(label=f"📥 Download {df_name.capitalize()} Data (CSV)", data=csv_data,
                                                   file_name=f"all_cities_{df_name}_data.csv", mime="text/csv", key=f"download_all_{df_name}_csv")
                            else:
                                st.caption(f"No {df_name} data to export as CSV.")
                
                else: # Specific city selected
                    filename_base = f"{export_city_sel_tab.lower().replace(' ', '_')}_environmental"
                    city_specific_export_data = {}
                    city_dfs_for_csv = {}

                    if export_city_sel_tab in climate_data:
                        city_specific_export_data['climate'] = climate_data[export_city_sel_tab]
                        # Reconstruct DataFrame for CSV for specific city's time-series
                        records = [{'date': d, **m} for d, m in climate_data[export_city_sel_tab].items()]
                        if records: city_dfs_for_csv['climate'] = pd.DataFrame(records)

                    if export_city_sel_tab in air_data:
                        city_specific_export_data['air'] = air_data[export_city_sel_tab]
                        records = [{'date': d, **m} for d, m in air_data[export_city_sel_tab].items()]
                        if records: city_dfs_for_csv['air'] = pd.DataFrame(records)

                    if export_city_sel_tab in soil_data:
                        city_specific_export_data['soil'] = soil_data[export_city_sel_tab]
                        if soil_data[export_city_sel_tab]: city_dfs_for_csv['soil'] = pd.DataFrame([soil_data[export_city_sel_tab]])
                    
                    if export_city_sel_tab in water_data:
                        city_specific_export_data['water'] = water_data[export_city_sel_tab]
                        if water_data[export_city_sel_tab]: city_dfs_for_csv['water'] = pd.DataFrame([water_data[export_city_sel_tab]])

                    if not city_specific_export_data:
                        st.warning(f"No data found for {export_city_sel_tab} to export.")
                    else:
                        if export_format_sel_tab == "JSON":
                            try:
                                json_data_city = json.dumps(city_specific_export_data, indent=4, cls=NpEncoder)
                                st.download_button(label=f"📥 Download {export_city_sel_tab} Data (JSON)", data=json_data_city,
                                                file_name=f"{filename_base}.json", mime="application/json", key="download_city_json")
                            except Exception as e:
                                st.error(f"Error generating JSON for {export_city_sel_tab}: {e}")
                        
                        elif export_format_sel_tab == "CSV":
                            if not city_dfs_for_csv:
                                st.warning(f"Could not prepare CSV data for {export_city_sel_tab}.")
                            else:
                                # For a single city, it might be better to combine into one CSV if schemas are simple enough,
                                # or offer separate files like for "All Data". For now, let's offer separate files.
                                st.info(f"For {export_city_sel_tab} in CSV format, individual files for each available data type will be provided below.")
                                for data_type_name, df_city_content in city_dfs_for_csv.items():
                                    if not df_city_content.empty:
                                        csv_city_data = df_city_content.to_csv(index=False).encode('utf-8')
                                        st.download_button(label=f"📥 Download {export_city_sel_tab} {data_type_name.capitalize()} Data (CSV)",
                                                           data=csv_city_data,
                                                           file_name=f"{filename_base}_{data_type_name}.csv",
                                                           mime="text/csv", key=f"download_city_{data_type_name}_csv")
                                    else:
                                         st.caption(f"No {data_type_name} data to export as CSV for {export_city_sel_tab}.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem 0;'>
    <p>🌿 Environment Impact Analyzer | Powered by Advanced Analytics & AI (Using Enhanced Sample Data)</p>
    <p>For sustainable decision-making. 
    AI features are optional: if <code>google-generativeai</code> is installed and <code>GOOGLE_API_KEY</code> is in <code>.streamlit/secrets.toml</code>, they will be enabled.</p>
</div>
""", unsafe_allow_html=True)