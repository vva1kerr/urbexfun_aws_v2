import streamlit as st
# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Satellite View")

# Import sidebar
from components.sidebar import show_sidebar

# Show the sidebar
show_sidebar()

import cv2
import requests
import numpy as np
import threading
from PIL import Image
import io
from src.database.db_utils import get_all_states, get_cities_in_state, get_city_info
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

st.title("Satellite View")

def project_with_scale(lat, lon, scale):
    """Mercator projection with scale"""
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y

def create_session_with_retries():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retries = Retry(
        total=5,  # number of retries
        backoff_factor=0.1,  # time factor between retries
        status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_tile(url, headers, channels):
    """Download a single map tile with retries"""
    session = create_session_with_retries()
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(arr, 1) if channels == 3 else cv2.imdecode(arr, -1)
        except Exception as e:
            if attempt == max_attempts - 1:  # Last attempt
                st.warning(f"Failed to download tile after {max_attempts} attempts: {url}")
                return None
            time.sleep(1)  # Wait before retrying

def get_satellite_image(lat1, lon1, lat2, lon2, zoom=12): 
    """Get satellite imagery for the specified bounds"""
    
    # Ensure correct coordinate ordering
    min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
    min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
    
    # Configuration
    tile_size = 256
    channels = 3
    url = 'https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'
    }
    
    scale = 1 << zoom
    
    # Find pixel and tile coordinates
    tl_proj_x, tl_proj_y = project_with_scale(max_lat, min_lon, scale)  # Top-left uses max_lat
    br_proj_x, br_proj_y = project_with_scale(min_lat, max_lon, scale)  # Bottom-right uses min_lat
    
    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)
    
    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)
    
    # Create image array
    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)
    
    # Create progress bar
    total_tiles = (br_tile_x - tl_tile_x + 1) * (br_tile_y - tl_tile_y + 1)
    progress_bar = st.progress(0)
    tiles_processed = 0
    
    # Download and stitch tiles
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            tile = download_tile(url.format(x=tile_x, y=tile_y, z=zoom), headers, channels)
            
            if tile is not None:
                # Calculate tile placement
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size
                
                # Define placement bounds
                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w + 1, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h + 1, br_rel_y)
                
                # Define crop bounds
                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)
                
                # Update progress
                tiles_processed += 1
                progress_percentage = int((tiles_processed / total_tiles) * 100)
                progress_bar.progress(progress_percentage)
                
                # Place tile in image
                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]
    
    # Clear progress bar
    progress_bar.empty()
    return img

# Initialize location_data
location_data = None

# Add this before the location input selection
st.write("### Select Location Method")
input_method = st.radio(
    "Choose input method:",
    ["City Selection", "Coordinate Bounds", "Single Point with Scale"]
)

# Add city selection logic
if input_method == "City Selection":
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("Select State", options=get_all_states())
    with col2:
        cities = get_cities_in_state(state)
        city = st.selectbox("Select City", options=cities)
        
    zoom = st.slider("Detail Level", 10, 20, 13,
        help="Zoom level for satellite imagery (higher = more detailed)")
    
    # Make button full width
    st.write("")  # Add spacing
    if st.button("Get Satellite Image", use_container_width=True):
        city_data = get_city_info(city, state)
        if city_data and city_data['latitude'] and city_data['longitude']:
            # Calculate bounds based on city's area
            area_miles = float(city_data['area_mile2'])
            # Rough conversion of square miles to degrees (approximately)
            degree_offset = min(max(np.sqrt(area_miles) * 0.02, 0.01), 0.2)
            
            location_data = {
                "bounds": (
                    city_data['latitude'] + degree_offset,  # max_lat (North)
                    city_data['longitude'] - degree_offset, # min_lon (West)
                    city_data['latitude'] - degree_offset,  # min_lat (South)
                    city_data['longitude'] + degree_offset  # max_lon (East)
                ),
                "zoom": zoom
            }
        else:
            st.error("No coordinate data available for this city.")

# Modify the existing location_method radio to an elif
elif input_method == "Coordinate Bounds":
    col1, col2 = st.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=-74.01)
        min_lat = st.number_input("Min Latitude", value=40.70)
    with col2:
        max_lon = st.number_input("Max Longitude", value=-73.95)
        max_lat = st.number_input("Max Latitude", value=40.75)
    
    zoom = st.slider("Detail Level", 10, 20, 13,
        help="Zoom level for satellite imagery (higher = more detailed)")
    
    if st.button("Get Satellite Image", use_container_width=True):
        location_data = {
            "bounds": (max_lat, min_lon, min_lat, max_lon),  # Order: North, West, South, East
            "zoom": zoom
        }

else:  # Single Point with Scale
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=40.7128)
        lon = st.number_input("Longitude", value=-74.0060)
    with col2:
        scale = st.slider("Area Size (degrees)", 0.01, 0.2, 0.05, 
            help="Size of the area around the point (larger = bigger area)")
        zoom = st.slider("Detail Level", 10, 20, 13,
            help="Zoom level for satellite imagery (higher = more detailed)")
    
    if st.button("Get Satellite Image", use_container_width=True):
        # Calculate bounds from center point and scale
        location_data = {
            "bounds": (
                lat + scale/2,  # max_lat (North)
                lon - scale/2,  # min_lon (West)
                lat - scale/2,  # min_lat (South)
                lon + scale/2   # max_lon (East)
            ),
            "zoom": zoom
        }

if location_data:
    try:
        with st.spinner("Fetching satellite imagery..."):
            image = get_satellite_image(
                location_data["bounds"][0],
                location_data["bounds"][1],
                location_data["bounds"][2],
                location_data["bounds"][3],
                location_data["zoom"]
            )
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display the image
            st.image(image_rgb, caption="Satellite View", use_container_width=True)
            
            # Show coordinates
            st.write("**Coordinates:**")
            st.write(f"Top-left: ({location_data['bounds'][0]:.4f}°N, {location_data['bounds'][1]:.4f}°W)")
            st.write(f"Bottom-right: ({location_data['bounds'][2]:.4f}°N, {location_data['bounds'][3]:.4f}°W)")
            
    except Exception as e:
        st.error(f"Error fetching satellite imagery: {str(e)}") 