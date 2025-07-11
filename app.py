import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# Set page configuration
st.set_page_config(
    page_title="İstanbul ve Çevresi İçin Yapay Zeka Tabanlı Deprem Erken Uyarı Sistemi",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5733;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498DB;
    }
    .warning-box {
        background-color: #FFCCCC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #FF5733;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #D6EAF8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #3498DB;
        margin: 1rem 0;
    }
    .safe-box {
        background-color: #D5F5E3;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #2ECC71;
        margin: 1rem 0;
    }
    .earthquake-list {
        height: 400px;
        overflow-y: auto;
        background-color: #F8F9F9;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh every 60 seconds
refresh_interval = st.sidebar.slider("Otomatik Yenileme (Saniye)", 30, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")

# Define Istanbul coordinates
ISTANBUL_COORDS = (41.0082, 28.9784)

# Define target cities and their coordinates
TARGET_CITIES = {
    "İstanbul": (41.0082, 28.9784),
    "Kocaeli": (40.7654, 29.9408),
    "Tekirdağ": (40.9781, 27.5126),
    "Sakarya": (40.7731, 30.3925),
    "Yalova": (40.6550, 29.2774),
    "Bursa": (40.1885, 29.0610)
}

# Istanbul districts and their coordinates
ISTANBUL_DISTRICTS = {
    "Adalar": (40.8760, 29.0878),
    "Arnavutköy": (41.1839, 28.7419),
    "Ataşehir": (40.9830, 29.1291),
    "Avcılar": (41.0204, 28.7187),
    "Bağcılar": (41.0378, 28.8500),
    "Bahçelievler": (41.0021, 28.8577),
    "Bakırköy": (40.9817, 28.8773),
    "Başakşehir": (41.0931, 28.8026),
    "Bayrampaşa": (41.0467, 28.8967),
    "Beşiktaş": (41.0434, 29.0086),
    "Beykoz": (41.1473, 29.0988),
    "Beylikdüzü": (41.0103, 28.6428),
    "Beyoğlu": (41.0366, 28.9735),
    "Büyükçekmece": (41.0195, 28.5933),
    "Çatalca": (41.1426, 28.4515),
    "Çekmeköy": (41.0330, 29.1872),
    "Esenler": (41.0437, 28.8763),
    "Esenyurt": (41.0290, 28.6728),
    "Eyüp": (41.0478, 28.9339),
    "Fatih": (41.0187, 28.9394),
    "Gaziosmanpaşa": (41.0680, 28.9097),
    "Güngören": (41.0178, 28.8898),
    "Kadıköy": (40.9926, 29.0233),
    "Kağıthane": (41.0784, 28.9833),
    "Kartal": (40.8884, 29.1872),
    "Küçükçekmece": (41.0015, 28.7981),
    "Maltepe": (40.9351, 29.1362),
    "Pendik": (40.8750, 29.2583),
    "Sancaktepe": (41.0006, 29.2266),
    "Sarıyer": (41.1693, 29.0557),
    "Silivri": (41.0731, 28.2464),
    "Sultanbeyli": (40.9650, 29.2652),
    "Sultangazi": (41.1066, 28.8679),
    "Şile": (41.1748, 29.6119),
    "Şişli": (41.0603, 28.9868),
    "Tuzla": (40.8156, 29.3009),
    "Ümraniye": (41.0161, 29.0964),
    "Üsküdar": (41.0284, 29.0258),
    "Zeytinburnu": (41.0070, 28.9000)
}
# Function to calculate risk level for Istanbul based on earthquake parameters
def calculate_risk_level(magnitude, depth, distance, time_since):
    # Base risk from magnitude
    if magnitude >= 7.0:
        base_risk = 5  # Very High
    elif magnitude >= 6.0:
        base_risk = 4  # High
    elif magnitude >= 5.0:
        base_risk = 3  # Moderate
    elif magnitude >= 4.0:
        base_risk = 2  # Low
    else:
        base_risk = 1  # Very Low
    
    # Adjust for depth (shallow earthquakes are more dangerous)
    if depth < 10:
        depth_factor = 1.5
    elif depth < 30:
        depth_factor = 1.2
    elif depth < 50:
        depth_factor = 1.0
    else:
        depth_factor = 0.8
    
    # Adjust for distance
    if distance < 50:
        distance_factor = 1.5
    elif distance < 100:
        distance_factor = 1.2
    elif distance < 200:
        distance_factor = 0.9
    else:
        distance_factor = 0.6
    
    # Adjust for time (more recent earthquakes might indicate active fault movement)
    if time_since.total_seconds() < 3600:  # Last hour
        time_factor = 1.3
    elif time_since.total_seconds() < 86400:  # Last day
        time_factor = 1.1
    elif time_since.total_seconds() < 604800:  # Last week
        time_factor = 0.9
    else:
        time_factor = 0.7
    
    # Calculate final risk score
    risk_score = base_risk * depth_factor * distance_factor * time_factor
    
    # Normalize to 1-5 scale
    risk_score = max(1, min(5, risk_score))
    
    return risk_score

# Function to estimate arrival time of seismic waves
def estimate_arrival_time(distance_km):
    # P-waves travel at approximately 6-8 km/s
    p_wave_speed = 7  # km/s
    
    # S-waves travel at approximately 3-4 km/s
    s_wave_speed = 3.5  # km/s
    
    p_wave_time = distance_km / p_wave_speed
    s_wave_time = distance_km / s_wave_speed
    
    return p_wave_time, s_wave_time

# Function to fetch earthquake data from Kandilli Observatory API
def fetch_kandilli_data():
    try:
        url = "http://www.koeri.boun.edu.tr/scripts/lst9.asp"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Parse the text response (custom format from Kandilli)
            content = response.text
            lines = content.split('\n')
            
            # Find where the earthquake data starts (after the header)
            start_idx = 0
            for i, line in enumerate(lines):
                if "-------------" in line:
                    start_idx = i + 1
                    break
            
            # Parse each earthquake entry
            earthquakes = []
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                
                try:
                    # Format is like: 2023.06.05 12:53:54 40.6877 27.5055 7.0 -.- 3.0 -.- SARKOYMARMARA_DENIZI (CANAKKALE) İlksel
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    
                    date = parts[0]
                    time_str = parts[1]
                    lat = float(parts[2])
                    lon = float(parts[3])
                    depth = float(parts[4])
                    magnitude = float(parts[6])
                    
                    # Extract location
                    loc_start = line.find('(')
                    loc_end = line.find(')')
                    if loc_start > 0 and loc_end > loc_start:
                        location = line[loc_start+1:loc_end]
                    else:
                        location = "Unknown"
                    
                    # Calculate distance to Istanbul
                    distance_to_istanbul = geodesic((lat, lon), ISTANBUL_COORDS).kilometers
                    
                    # Parse datetime
                    dt_str = f"{date} {time_str}"
                    dt = datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S")
                    
                    earthquakes.append({
                        'date': dt,
                        'latitude': lat,
                        'longitude': lon,
                        'depth': depth,
                        'magnitude': magnitude,
                        'location': location,
                        'distance_to_istanbul': distance_to_istanbul
                    })
                except Exception as e:
                    # Skip entries that can't be parsed
                    continue
            
            # Sort by date (newest first)
            earthquakes.sort(key=lambda x: x['date'], reverse=True)
            return earthquakes
        else:
            st.error("Failed to fetch data from Kandilli Observatory.")
            return []
    except Exception as e:
        st.error(f"Error fetching earthquake data: {e}")
        return []

# Alternative function to fetch from USGS if Kandilli fails
def fetch_usgs_data():
    try:
        # Get earthquakes from the past 30 days with magnitude > 2.5 near Turkey
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {
            "format": "geojson",
            "starttime": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "endtime": datetime.now().strftime("%Y-%m-%d"),
            "minmagnitude": 2.5,
            "latitude": ISTANBUL_COORDS[0],
            "longitude": ISTANBUL_COORDS[1],
            "maxradiuskm": 500
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            earthquakes = []
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                # Extract data
                magnitude = props['mag']
                dt = datetime.fromtimestamp(props['time'] / 1000)
                location = props['place']
                depth = coords[2]
                longitude = coords[0]
                latitude = coords[1]
                
                # Calculate distance to Istanbul
                distance_to_istanbul = geodesic((latitude, longitude), ISTANBUL_COORDS).kilometers
                
                earthquakes.append({
                    'date': dt,
                    'latitude': latitude,
                    'longitude': longitude,
                    'depth': depth,
                    'magnitude': magnitude,
                    'location': location,
                    'distance_to_istanbul': distance_to_istanbul
                })
            
            # Sort by date (newest first)
            earthquakes.sort(key=lambda x: x['date'], reverse=True)
            return earthquakes
        else:
            st.error("Failed to fetch data from USGS.")
            return []
    except Exception as e:
        st.error(f"Error fetching USGS earthquake data: {e}")
        return []

# Main function to get earthquake data (tries Kandilli first, falls back to USGS)
@st.cache_data(ttl=60)
def get_earthquake_data():
    kandilli_data = fetch_kandilli_data()
    if kandilli_data:
        return kandilli_data
    else:
        return fetch_usgs_data()

# Sidebar for filters and settings
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/58/Earthquake_hazard_symbol.svg", width=100)
    st.title("Filtreler ve Ayarlar")
    
    st.subheader("Deprem Filtreleri")
    min_magnitude = st.slider("Minimum Büyüklük", 0.0, 10.0, 3.0, 0.1)
    
    max_distance = st.slider("İstanbul'a Maksimum Uzaklık (km)", 50, 1000, 500)
    
    days_back = st.slider("Son Kaç Gün", 1, 30, 7)
    min_date = datetime.now() - timedelta(days=days_back)
    
    st.subheader("Bildirim Ayarları")
    notification_threshold = st.slider("Bildirim için Minimum Büyüklük", 3.0, 7.0, 4.5, 0.1)
    
    # Districts selection for Istanbul
    st.subheader("İstanbul İlçe Seçimi")
    selected_districts = st.multiselect(
        "İlçeler",
        list(ISTANBUL_DISTRICTS.keys()),
        default=["Kadıköy", "Fatih", "Beşiktaş", "Üsküdar"]
    )
    
    # Add informational box in sidebar
    st.info("Bu uygulama, Kandilli Rasathanesi ve USGS verilerini kullanarak İstanbul ve çevresi için deprem risk analizi yapar. Veriler her {} saniyede bir güncellenir.".format(refresh_interval))

# Get earthquake data
earthquakes = get_earthquake_data()

# Apply filters
filtered_earthquakes = [
    eq for eq in earthquakes 
    if eq['magnitude'] >= min_magnitude 
    and eq['date'] >= min_date
    and eq['distance_to_istanbul'] <= max_distance
]
# Main content
st.markdown("<h1 class='main-header'>İstanbul ve Çevresi İçin Yapay Zeka Tabanlı Deprem Erken Uyarı Sistemi</h1>", unsafe_allow_html=True)

# Recent strong earthquake alert (if any)
recent_strong_earthquakes = [
    eq for eq in earthquakes 
    if eq['magnitude'] >= notification_threshold 
    and (datetime.now() - eq['date']).total_seconds() < 3600 * 24  # Last 24 hours
    and eq['distance_to_istanbul'] <= 300  # Within 300km of Istanbul
]

if recent_strong_earthquakes:
    st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
    st.markdown("### ⚠️ DİKKAT: Son 24 Saat İçinde Güçlü Deprem!")
    
    for eq in recent_strong_earthquakes[:3]:  # Show up to 3 recent strong earthquakes
        # Calculate estimated arrival time to Istanbul
        dist_km = eq['distance_to_istanbul']
        p_time, s_time = estimate_arrival_time(dist_km)
        
        st.markdown(f"""
        **{eq['date'].strftime('%d.%m.%Y %H:%M:%S')}** - **{eq['magnitude']:.1f}** büyüklüğünde deprem
        - Konum: {eq['location']}
        - İstanbul'a uzaklık: {dist_km:.1f} km
        - Derinlik: {eq['depth']} km
        - P-dalgası varış süresi: ~{p_time:.1f} saniye
        - S-dalgası varış süresi: ~{s_time:.1f} saniye
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='safe-box'>", unsafe_allow_html=True)
    st.markdown("### ✅ Şu an için İstanbul'u tehdit eden büyük bir deprem bulunmamaktadır.")
    st.markdown("Son deprem verilerine göre şu anda İstanbul ve çevresi için acil bir tehdit tespit edilmedi.")
    st.markdown("</div>", unsafe_allow_html=True)

# Create two columns for the dashboard
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='sub-header'>Deprem Haritası</h2>", unsafe_allow_html=True)
    
    # Create a map centered on Istanbul
    m = folium.Map(location=ISTANBUL_COORDS, zoom_start=7)
    
    # Add markers for filtered earthquakes
    for eq in filtered_earthquakes:
        # Determine color based on magnitude
        if eq['magnitude'] >= 5.0:
            color = 'red'
        elif eq['magnitude'] >= 4.0:
            color = 'orange'
        else:
            color = 'green'
        
        # Create popup content
        popup_content = f"""
        <strong>Tarih:</strong> {eq['date'].strftime('%d.%m.%Y %H:%M:%S')}<br>
        <strong>Büyüklük:</strong> {eq['magnitude']:.1f}<br>
        <strong>Derinlik:</strong> {eq['depth']} km<br>
        <strong>Konum:</strong> {eq['location']}<br>
        <strong>İstanbul'a uzaklık:</strong> {eq['distance_to_istanbul']:.1f} km
        """
        
        # Add marker
        folium.CircleMarker(
            location=[eq['latitude'], eq['longitude']],
            radius=eq['magnitude'] * 2,  # Size based on magnitude
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(m)
    
    # Add markers for target cities
    for city, coords in TARGET_CITIES.items():
        folium.Marker(
            location=coords,
            popup=city,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Add markers for selected Istanbul districts
    for district in selected_districts:
        if district in ISTANBUL_DISTRICTS:
            folium.Marker(
                location=ISTANBUL_DISTRICTS[district],
                popup=f"İstanbul - {district}",
                icon=folium.Icon(color='purple', icon='home')
            ).add_to(m)
    
    # Display the map
    folium_static(m)
    
    # Add legend for the map
    st.markdown("""
    **Harita Lejandı:**
    - 🔴 Kırmızı: 5.0+ büyüklüğünde depremler
    - 🟠 Turuncu: 4.0-4.9 büyüklüğünde depremler
    - 🟢 Yeşil: 3.0-3.9 büyüklüğünde depremler
    - 🔵 Mavi: Şehir merkezleri
    - 🟣 Mor: Seçilen İstanbul ilçeleri
    """)

with col2:
    st.markdown("<h2 class='sub-header'>Son Depremler</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='earthquake-list'>", unsafe_allow_html=True)
    for eq in filtered_earthquakes[:30]:  # Show top 30 earthquakes
        # Calculate time difference
        time_diff = datetime.now() - eq['date']
        if time_diff.total_seconds() < 3600:
            time_str = f"{int(time_diff.total_seconds() / 60)} dakika önce"
        elif time_diff.total_seconds() < 86400:
            time_str = f"{int(time_diff.total_seconds() / 3600)} saat önce"
        else:
            time_str = f"{int(time_diff.total_seconds() / 86400)} gün önce"
        
        # Determine color based on magnitude
        if eq['magnitude'] >= 5.0:
            mag_color = "🔴"
        elif eq['magnitude'] >= 4.0:
            mag_color = "🟠"
        else:
            mag_color = "🟢"
        
        st.markdown(f"""
        {mag_color} **{eq['magnitude']:.1f}** | {eq['date'].strftime('%d.%m.%Y %H:%M')} | {time_str}  
        📍 {eq['location']} | 🧭 {eq['distance_to_istanbul']:.1f} km | 🕳️ {eq['depth']} km
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk assessment for Istanbul
    st.markdown("<h2 class='sub-header'>İstanbul için Risk Değerlendirmesi</h2>", unsafe_allow_html=True)
    
    # Calculate current risk level based on recent earthquakes
    risk_scores = []
    for eq in earthquakes[:50]:  # Consider the 50 most recent earthquakes
        time_since = datetime.now() - eq['date']
        if time_since.total_seconds() <= 604800:  # Last week
            risk = calculate_risk_level(
                eq['magnitude'], 
                eq['depth'], 
                eq['distance_to_istanbul'], 
                time_since
            )
            risk_scores.append(risk)
    
    # Calculate overall risk (if we have data)
    if risk_scores:
        overall_risk = np.mean(risk_scores)
        
        # Display risk meter
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Seviyesi"},
            gauge = {
                'axis': {'range': [0, 5], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1], 'color': '#2ECC71'},
                    {'range': [1, 2], 'color': '#82E0AA'},
                    {'range': [2, 3], 'color': '#F7DC6F'},
                    {'range': [3, 4], 'color': '#F5B041'},
                    {'range': [4, 5], 'color': '#E74C3C'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation
        if overall_risk < 1.5:
            risk_text = "Çok Düşük Risk"
            risk_description = "Şu anda İstanbul ve çevresi için deprem riski çok düşük seviyededir."
        elif overall_risk < 2.5:
            risk_text = "Düşük Risk"
            risk_description = "İstanbul ve çevresi için deprem riski düşük seviyededir. Temel önlemler alınmalıdır."
        elif overall_risk < 3.5:
            risk_text = "Orta Risk"
            risk_description = "İstanbul ve çevresi için deprem riski orta seviyededir. Önlem ve hazırlıklarınızı gözden geçirin."
        elif overall_risk < 4.5:
            risk_text = "Yüksek Risk"
            risk_description = "İstanbul ve çevresi için deprem riski yüksek seviyededir. Dikkatli olun ve tüm önlemleri alın."
        else:
            risk_text = "Çok Yüksek Risk"
            risk_description = "İstanbul ve çevresi için deprem riski çok yüksek seviyededir. Acil durum hazırlıklarınızı kontrol edin."
        
        st.markdown(f"**Risk Değerlendirmesi:** {risk_text}")
        st.markdown(risk_description)
    else:
        st.warning("Risk değerlendirmesi için yeterli veri bulunmamaktadır.")
# Add tabs for additional information
tab1, tab2, tab3 = st.tabs(["İstatistikler", "Güvenlik Rehberi", "Güvenli Toplanma Alanları"])

with tab1:
    st.markdown("<h3 class='sub-header'>Deprem İstatistikleri</h3>", unsafe_allow_html=True)
    
    # Convert to dataframe for analysis
    df = pd.DataFrame(earthquakes)
    
    if not df.empty:
        # Display some statistics
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            # Magnitude distribution
            fig = px.histogram(
                df, 
                x="magnitude", 
                nbins=20, 
                title="Deprem Büyüklük Dağılımı",
                color_discrete_sequence=['#3498DB']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_stats2:
            # Depth distribution
            fig = px.histogram(
                df, 
                x="depth", 
                nbins=20, 
                title="Deprem Derinlik Dağılımı",
                color_discrete_sequence=['#2ECC71']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series of earthquakes
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Resample by day and count
            daily_counts = df.resample('D').size()
            
            fig = px.line(
                daily_counts, 
                title="Günlük Deprem Sayısı",
                labels={'value': 'Deprem Sayısı', 'date': 'Tarih'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Magnitude vs. Depth scatter plot
        fig = px.scatter(
            df, 
            x="magnitude", 
            y="depth",
            title="Büyüklük ve Derinlik İlişkisi",
            labels={'magnitude': 'Büyüklük', 'depth': 'Derinlik (km)'},
            color="magnitude",
            size="magnitude",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        fig.update_yaxes(autorange="reversed")  # Reverse y-axis so smaller depth values are at the top
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("İstatistikler için veri bulunmamaktadır.")

with tab2:
    st.markdown("<h3 class='sub-header'>Deprem Güvenlik Rehberi</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Deprem Öncesi Hazırlık
    
    1. **Acil Durum Planı Yapın**
       - Ailenizle buluşma noktası belirleyin
       - Önemli numaraları kaydedin
       - Alternatif iletişim yolları belirleyin
    
    2. **Acil Durum Çantası Hazırlayın**
       - Su ve uzun süre dayanabilecek gıdalar
       - İlk yardım malzemeleri
       - El feneri ve piller
       - Düdük
       - Önemli evrakların kopyaları
       - Nakit para
       - Şarj edilebilir powerbank
    
    3. **Evinizi Güvenli Hale Getirin**
       - Ağır eşyaları sabitleyin
       - Kaçış yollarını belirleyin
       - Gaz, su ve elektrik vanalarının yerlerini öğrenin
    
    ### Deprem Anında
    
    1. **İçerideyseniz**
       - **ÇÖK - KAPAN - TUTUN** prensibini uygulayın
       - Sağlam bir masa altına girin
       - Başınızı ve boynunuzu koruyun
       - Pencerelerden uzak durun
       - Asansör kullanmayın
    
    2. **Dışarıdaysanız**
       - Açık alana çıkın
       - Binalardan, ağaçlardan, elektrik tellerinden uzak durun
    
    3. **Araçtaysanız**
       - Güvenli bir şekilde durun
       - Köprülerden, viyadüklerden, tünellerden uzaklaşın
       - Araçta kalın ve aracın içinde ÇÖK-KAPAN-TUTUN yapın
    
    ### Deprem Sonrası
    
    1. **Kendinizi ve Çevrenizdekileri Kontrol Edin**
       - Yaralanma durumunda ilk yardım uygulayın
       - Sakin kalın ve paniklememeye çalışın
    
    2. **Güvenlik Önlemleri**
       - Gaz, su ve elektriği kapatın
       - Hasarlı binalardan uzak durun
       - Artçı sarsıntılara karşı dikkatli olun
    
    3. **Yardım ve İletişim**
       - Acil durumlarda 112'yi arayın
       - Sosyal medya yerine kısa mesaj kullanın (SMS)
       - Resmi bilgileri dinleyin (AFAD, Kandilli)
    """)
    
    st.info("""
    **Acil Durum İletişim Bilgileri:**
    - AFAD: 122
    - Acil Çağrı: 112
    - İtfaiye: 110
    - Polis: 155
    - Jandarma: 156
    """)

with tab3:
    st.markdown("<h3 class='sub-header'>Güvenli Toplanma Alanları</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    ### İstanbul'daki Güvenli Toplanma Alanları
    
    İstanbul'da AFAD tarafından belirlenmiş resmi toplanma alanları bulunmaktadır. Bu alanlar deprem sonrası güvenli bölgeler olarak tasarlanmıştır.
    """)
    
    # Create a map for meeting points
    st.markdown("#### Harita Üzerinde Toplanma Alanları")
    
    # Sample meeting points data (would ideally come from AFAD API)
    meeting_points = {
        "Kadıköy - Fenerbahçe Parkı": (40.9697, 29.0367),
        "Beşiktaş - İnönü Stadı Çevresi": (41.0421, 29.0148),
        "Fatih - Yenikapı Etkinlik Alanı": (41.0021, 28.9744),
        "Üsküdar - Doğancılar Parkı": (41.0265, 29.0152),
        "Bakırköy - Botanik Parkı": (40.9799, 28.8740),
        "Maltepe - Sahil Alanı": (40.9343, 29.1235),
        "Beylikdüzü - Yaşam Vadisi": (41.0017, 28.6394),
        "Ataşehir - Atatürk Parkı": (40.9847, 29.1272),
        "Beykoz - Çubuklu Sahili": (41.1058, 29.0835),
        "Sarıyer - Maslak Atatürk Oto Sanayi": (41.1150, 29.0117),
        "Pendik - Sahil Alanı": (40.8739, 29.2356),
        "Büyükçekmece - Sahil Alanı": (41.0224, 28.5941),
    }
    
    # Create map
    m = folium.Map(location=ISTANBUL_COORDS, zoom_start=10)
    
    # Add markers for meeting points
    for name, coords in meeting_points.items():
        folium.Marker(
            location=coords,
            popup=name,
            icon=folium.Icon(color='green', icon='flag')
        ).add_to(m)
    
    # Display the map
    folium_static(m)
    
    st.markdown("""
    #### Nasıl Toplanma Alanı Bulunur?
    
    1. **e-Devlet**: e-Devlet üzerinden adresinize en yakın toplanma alanını öğrenebilirsiniz.
    2. **AFAD Web Sitesi**: AFAD'ın resmi web sitesinden toplanma alanlarını kontrol edebilirsiniz.
    3. **Belediye Web Siteleri**: İlçe belediyelerinin web sitelerinde toplanma alanları listelenmiştir.
    4. **"AFAD Acil" Mobil Uygulaması**: Bu uygulama üzerinden size en yakın toplanma alanını görebilirsiniz.
    
    > **NOT:** Yukarıdaki haritada gösterilen toplanma alanları örnek niteliğindedir. Gerçek ve güncel toplanma alanlarını AFAD'ın resmi kaynaklarından kontrol ediniz.
    """)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("© 2025 İstanbul Deprem Erken Uyarı Sistemi | Bu uygulama sadece bilgilendirme amaçlıdır ve resmi bir acil durum sistemi değildir.")
st.markdown("</div>", unsafe_allow_html=True)
