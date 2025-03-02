import streamlit as st
import os
from PIL import Image
from components.sidebar import show_sidebar  # Import the sidebar component

# Must be first Streamlit command
st.set_page_config(layout="wide", page_title="Urban Explorer")

# Show the shared sidebar
show_sidebar()

# Main content
st.title("Urban Explorer (UrbEx)")
st.markdown("""
## About Urban Exploration
Urban exploration (often shortened to UrbEx) is the practice of exploring human-made structures, usually abandoned ruins or hidden components of the built environment. UrbEx often has strong ties to photography and historical documentation, helping preserve the memory of forgotten places.

**Important Notice:** This tool is for educational and research purposes only. Always:
- Obtain proper permissions before visiting any location
- Follow local laws and regulations
- Prioritize safety and respect for property
- Leave everything as you found it
- Take only photographs, leave only footprints

## Application Features

### 1. Map Search 🗺️
*Navigate to the "Map" page to:*
- Search for any city in the United States
- View detailed city information including:
  - Population and density statistics
  - Geographic boundaries
  - Current weather conditions and alerts
- Interactive map with adjustable zoom levels
- Precise coordinate information

### 2. Topography Viewer 🏔️
*Navigate to the "Graph" page to:*
- View detailed elevation data for North America
  - Available range: Longitude (-180° to -10°), Latitude (10° to 80°)
- Choose between two input methods:
  - Coordinate bounds (direct area selection)
  - Single point with scale (radius-based selection)
- View multiple visualizations:
  - Digital Elevation Model (DEM)
  - Ridge line plots
  - 3D terrain visualization
- Compare full region vs. selected area

### 3. Satellite View 🛰️
*Navigate to the "Satellite" page to:*
- Access high-resolution satellite imagery
- Choose your area of interest using:
  - Precise coordinate bounds
  - Center point with adjustable scale
- Control image detail with zoom levels
- Perfect for:
  - Location scouting
  - Terrain analysis
  - Route planning

## How to Use

1. Start with the **Map Search** to find your area of interest
2. Use the **Topography Viewer** to understand the terrain
   - Great for identifying interesting geological features
   - Helps plan routes and assess accessibility
3. Switch to **Satellite View** for detailed ground-level planning
   - Verify access points
   - Identify structures and features
   - Assess current conditions

## Tips for Best Results

- **Map Search:** Use this to get exact coordinates for any US city
- **Topography:** Start with a larger area and then zoom in to areas of interest
- **Satellite:** Higher zoom levels (15-20) show the most detail but cover less area

## Data Sources
- Elevation data: USGS Digital Elevation Models
- Satellite imagery: Google Maps
- Weather data: OpenWeatherMap API
- City data: US Census Bureau (2024 projections)
- Topography .tif files: European Space Agency
""")

# Footer
st.markdown("---")
st.markdown("""
### About
This application combines various mapping and terrain analysis tools to help you visualize and understand topographical data.

### Getting Started
1. Select a tool from the sidebar
2. Enter location information
3. View and analyze the results

### Need Help?
Check out the documentation or contact support for assistance.
""")

# Add a footer with version info and disclaimer
st.markdown("---")
st.markdown("""
<small>Version 1.0.0 | For educational and research purposes only | Not for commercial use</small>
""", unsafe_allow_html=True) 