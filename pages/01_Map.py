import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="Map Search App", layout="wide")

# Import sidebar
from components.sidebar import show_sidebar

# Show the sidebar
show_sidebar()

# Rest of the imports
import folium
from streamlit_folium import st_folium
import geocoder
# import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
from src.get_coords.point_to_bounds import get_bounds_from_point
from src.get_coords.calculate_zoom import calculate_zoom_level
from src.weather.get_weather import get_weather_data
from src.database.db_utils import (
    get_all_states, 
    get_cities_in_state, 
    get_city_info,
    get_city_zipcodes,
    get_city_ips
)

# Cache the database queries
@st.cache_data
def load_states():
    return get_all_states()

@st.cache_data
def load_cities(state):
    return get_cities_in_state(state)

@st.cache_data
def load_city_info(city, state):
    return get_city_info(city, state)

@st.cache_data
def create_map(lat, lon, zoom, city=None, state=None, bounds=None):
    """Create a folium map with the given parameters"""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom
    )
    
    if city and state:
        folium.Marker(
            [lat, lon],
            popup=f"{city}, {state}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    
    if bounds and isinstance(bounds, dict) and all(k in bounds for k in ['bounds']):
        try:
            bound_coords = bounds['bounds']
            if all(k in bound_coords for k in ['south', 'west', 'north', 'east']):
                m.fit_bounds([
                    [bound_coords['south'], bound_coords['west']], 
                    [bound_coords['north'], bound_coords['east']]
                ])
        except Exception as e:
            st.write(f"Debug: Bounds error - {str(e)}")
            st.write(f"Debug: Bounds structure - {bounds}")
    
    return m

# Initialize session state
if 'map_data' not in st.session_state:
    st.session_state.map_data = {
        'lat': 40.7128,  # Default to NYC
        'lon': -74.0060,
        'zoom': 10,
        'm': None,
        'bounds': None,
        'city_info': None,
        'location': None,
        'last_search': None  # Track last search to prevent unnecessary updates
    }

if 'map_key' not in st.session_state:
    st.session_state.map_key = 0

try:
    # Load states
    sorted_states = load_states()
    if not sorted_states:
        st.error("No states found in database")
        st.stop()
        
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = sorted_states[0]

    # Load cities
    if 'city_options' not in st.session_state:
        st.session_state.city_options = load_cities(st.session_state.selected_state)

    if 'selected_city' not in st.session_state:
        st.session_state.selected_city = st.session_state.city_options[0]

    # Title and layout
    st.title("Map Search Application")

    # State and city selection
    col1, col2 = st.columns(2)
    state = col1.selectbox("Select State", options=sorted_states, key='state_select')
    cities_in_state = load_cities(state)
    if not cities_in_state:
        st.error(f"No cities found for state: {state}")
        st.stop()
    city = col2.selectbox("Select City", options=cities_in_state, key='city_select')

    # Search button
    search_key = f"{city}_{state}"  # Create a unique key for the current selection
    submitted = st.button("Search", type="primary", use_container_width=True)

    # Map container
    map_container = st.container()
    with map_container:

        if submitted and search_key != st.session_state.map_data.get('last_search'):
            st.session_state.map_data['last_search'] = search_key
            st.session_state.map_key += 1  # Force map refresh
            
            try:
                # Get city data from database
                city_data = load_city_info(city, state)
                if city_data:
                    area = float(city_data['area_mile2'])
                    
                    # Get coordinates from geocoding
                    location = geocoder.osm(
                        f"{city}, {state}, USA", 
                        headers={'User-Agent': 'my-app/1.0'},
                        timeout=10
                    )
                    
                    if location.ok:
                        bounds = get_bounds_from_point(location.lat, location.lng, area)
                        
                        st.session_state.map_data.update({
                            'lat': location.lat,
                            'lon': location.lng,
                            'location': location,
                            'city_info': city_data,
                            'bounds': bounds,
                            'zoom': calculate_zoom_level(bounds) if bounds else 10
                        })
                        
                        # Create a folium map
                        current_map = create_map(
                            lat=location.lat,
                            lon=location.lng,
                            zoom=st.session_state.map_data['zoom'],
                            city=city,
                            state=state,
                            bounds=bounds
                        )
                        
                        st.session_state.map_data['m'] = current_map
                        
                    else:
                        st.error(f"Location not found for {city}, {state}")

                        st.session_state.map_data['m'] = create_map(
                            lat=st.session_state.map_data['lat'],
                            lon=st.session_state.map_data['lon'],
                            zoom=st.session_state.map_data['zoom']
                        )
                else:
                    st.error(f"No data found for {city}, {state}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write(f"Debug: Exception details - {type(e).__name__}")
                st.write(f"Debug: Full error - {str(e)}")
                st.session_state.map_data['m'] = create_map(
                    lat=st.session_state.map_data['lat'],
                    lon=st.session_state.map_data['lon'],
                    zoom=st.session_state.map_data['zoom']
                )
        
        # If no map exists yet, create default map
        if st.session_state.map_data['m'] is None:
            st.write("Debug: Creating initial default map")
            st.session_state.map_data['m'] = create_map(
                lat=st.session_state.map_data['lat'],
                lon=st.session_state.map_data['lon'],
                zoom=st.session_state.map_data['zoom']
            )

        # Display the map with dynamic height
        map_data = st_folium(
            st.session_state.map_data['m'],
            height=600,
            width="100%",
            key=f"map_{st.session_state.map_key}",
            returned_objects=[]
        )
        st.write("Debug: Map display complete")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.write(f"Debug: Application exception details - {type(e).__name__}")

# Modify the information panel section to include zip codes and IPs
if (st.session_state.map_data['location'] is not None and 
    st.session_state.map_data['city_info'] is not None):
    
    # Get weather data
    weather_info = get_weather_data(
        st.session_state.map_data['lat'],
        st.session_state.map_data['lon']
    )
    
    # Create tabs for different types of information
    tab1, tab2, tab3 = st.tabs(["Basic Info", "Zip Codes", "IP Addresses"])
    
    with tab1:
        # Create four columns for basic information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("Location")
            st.write("**Coordinates:**")
            st.write(f"Latitude: {st.session_state.map_data['lat']:.4f}°N")
            st.write(f"Longitude: {st.session_state.map_data['lon']:.4f}°W")
        
        with col2:
            st.subheader("City Information")
            st.write(f"Population (2024): {st.session_state.map_data['city_info']['population_2024']:,}")
            st.write(f"Density: {st.session_state.map_data['city_info']['density_per_mile2']:,.0f} per sq mile")
            st.write(f"Area: {st.session_state.map_data['city_info']['area_mile2']:.2f} sq miles")
        
        with col3:
            if st.session_state.map_data['bounds']:
                st.subheader("Bounding Box")
                bounds = st.session_state.map_data['bounds']
                st.write(f"North: {bounds['bounds']['north']:.4f}°")
                st.write(f"South: {bounds['bounds']['south']:.4f}°")
                st.write(f"East: {bounds['bounds']['east']:.4f}°")
                st.write(f"West: {bounds['bounds']['west']:.4f}°")
        
        with col4:
            st.subheader("Current Weather")
            if 'error' in weather_info:
                st.error(weather_info['error'])
            else:
                st.write(f"**Temperature:** {weather_info['temperature']:.1f}°F")
                st.write(f"**Feels Like:** {weather_info['feels_like']:.1f}°F")
                st.write(f"**Humidity:** {weather_info['humidity']}%")
                st.write(f"**Wind Speed:** {weather_info['wind_speed']} mph")
                st.write(f"**Conditions:** {weather_info['description']}")
                
                if weather_info['alerts']:
                    st.warning("⚠️ Weather Alerts")
                    for alert in weather_info['alerts']:
                        with st.expander(f"Alert: {alert['event']}"):
                            st.write(alert['description'])
    
    with tab2:
        st.subheader(f"Zip Codes for {city}, {state}")
        zipcodes = get_city_zipcodes(city, state)
        if zipcodes:
            # Create a grid layout for zip codes
            cols = st.columns(5)  # Display in 5 columns
            for idx, zipcode in enumerate(zipcodes):
                cols[idx % 5].write(zipcode)
        else:
            st.info("No zip codes found for this city.")
            
    with tab3:
        st.subheader(f"IP Addresses for {city}, {state}")
        ip_addresses = get_city_ips(city, state)
        if ip_addresses:
            # Display IP addresses in a table
            ip_df = pd.DataFrame(ip_addresses, columns=['IP Address'])
            st.dataframe(ip_df, hide_index=True)
        else:
            st.info("No IP addresses found for this city.") 