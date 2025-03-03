import streamlit as st
# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Topography Viewer")

# Import sidebar
from components.sidebar import show_sidebar

# Show the sidebar
show_sidebar()

import rasterio
from rasterio.plot import show
from rasterio.windows import Window
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.mask import mask
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from PIL import Image
import numpy as np
import pandas as pd
import os
from ridge_map import RidgeMap
import plotly.graph_objects as go
from src.database.db_utils import get_all_states, get_cities_in_state, get_city_info
from src.cloud.s3_utils import S3Handler
import gc
import psutil
from dotenv import load_dotenv

# Memory management functions
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_threshold(threshold_mb=1600):
    """Check if memory usage is above threshold"""
    current_usage = get_memory_usage()
    if current_usage > threshold_mb:
        st.warning(f"High memory usage detected: {current_usage:.1f}MB. Try reducing data size.")
        return True
    return False

# Load environment variables
load_dotenv()

# Initialize S3 handler at the top of the file, after imports
s3_handler = S3Handler(
    bucket_name="urbexfun",  # This is just a placeholder now since we're using mounted filesystem
    region_name="us-east-2"   # This is just a placeholder now since we're using mounted filesystem
)

st.title("Topography Viewer")

# Add memory usage indicator
memory_usage = get_memory_usage()
st.sidebar.write(f"Current Memory Usage: {memory_usage:.1f}MB")

# Initialize location_data
location_data = None

# Create three equal-width columns
col1, col2, col3 = st.columns(3)
# Load and resize images to be the same size
target_width = 400  # You can adjust this value
target_height = 300  # Set a fixed height for all images
# Function to resize image to fixed dimensions
def load_and_resize(image_path, width, height):
    img = Image.open(image_path)
    return img.resize((width, height))
# Display images with use_container_width=True to make them fill their columns
with col1:
    img1 = load_and_resize('images/left.png', target_width, target_height)
    st.image(img1, use_container_width=True)
with col2:
    img2 = load_and_resize('images/right.png', target_width, target_height)
    st.image(img2, use_container_width=True)
with col3:
    img3 = load_and_resize('images/bottom.png', target_width, target_height)
    st.image(img3, use_container_width=True)

    
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
    
    # Make button full width
    st.write("")  # Add spacing
    if st.button("Get City Data", use_container_width=True):
        city_data = get_city_info(city, state)
        if city_data and city_data['latitude'] and city_data['longitude']:
            location_data = {
                "type": "bounds",
                "bounds": (
                    city_data['longitude'] - 0.1,  # min_lon
                    city_data['latitude'] - 0.1,   # min_lat
                    city_data['longitude'] + 0.1,  # max_lon
                    city_data['latitude'] + 0.1    # max_lat
                )
            }
        else:
            st.error("No coordinate data available for this city.")

# Modify the existing location_method radio to an elif
elif input_method == "Coordinate Bounds":
    col1, col2 = st.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=-180.0)
        min_lat = st.number_input("Min Latitude", value=10.0)
    with col2:
        max_lon = st.number_input("Max Longitude", value=0.01)
        max_lat = st.number_input("Max Latitude", value=89.99)
    
    # Make button full width
    st.write("")  # Add spacing
    if st.button("Submit Coordinates", use_container_width=True):
        location_data = {
            "type": "bounds",
            "bounds": (min_lon, min_lat, max_lon, max_lat)
        }

else:  # Single Point with Scale
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=40.73)
        lon = st.number_input("Longitude", value=-73.93)
    with col2:
        scale = st.slider("Zoom scale (degrees)", 0.1, 5.0, 1.0)
    
    if st.button("Submit Point", use_container_width=True):
        # Calculate view bounds first
        view_bounds = (
            lon - scale/2,  # min_lon
            lat - scale/2,  # min_lat
            lon + scale/2,  # max_lon
            lat + scale/2   # max_lat
        )
        
        # Get TIFF file based on view bounds center point
        tiff_path, downsample_factor = s3_handler.get_tiff_path(lat, lon)
        
        if not tiff_path:
            st.error("Could not find TIFF file for the given coordinates")
            st.stop()
            
        # Create location data with both point and bounds
        location_data = {
            "type": "bounds",
            "bounds": view_bounds,
            "point": (lat, lon),
            "tiff_path": tiff_path
        }
        
        # Debug output
        st.write("Debug Info:", {
            "Mount Point": "/home/ubuntu/s3bucket",
            "Input Point": f"({lat}°N, {lon}°W)",
            "Scale": f"{scale}°",
            "Selected TIFF": tiff_path,
            "View Bounds": view_bounds,
            "Downsample Factor": downsample_factor
        })

# Add this helper function near the top of the file
def clamp_bounds(requested_bounds, tiff_bounds):
    """
    Clamp requested bounds to stay within TIFF file bounds.
    Returns tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    # Ensure the order is correct (min_lon, min_lat, max_lon, max_lat)
    min_lon = max(min(requested_bounds[0], requested_bounds[2]), tiff_bounds.left)
    max_lon = min(max(requested_bounds[0], requested_bounds[2]), tiff_bounds.right)
    min_lat = max(min(requested_bounds[1], requested_bounds[3]), tiff_bounds.bottom)
    max_lat = min(max(requested_bounds[1], requested_bounds[3]), tiff_bounds.top)
    
    return (min_lon, min_lat, max_lon, max_lat)

@st.cache_data
def load_and_downsample_tiff(tiff_path, downsample_factor=4):
    """Load TIFF file with downsampling to manage memory"""
    try:
        # Check memory before loading
        current_memory = get_memory_usage()
        st.write(f"Memory usage before loading: {current_memory:.1f}MB")
        
        with rasterio.open(tiff_path) as src:
            # Read metadata
            width = src.width
            height = src.height
            
            # Calculate window size for downsampling
            window_width = width // downsample_factor
            window_height = height // downsample_factor
            
            # Read downsampled data with memory optimization
            data = src.read(
                1,
                out_shape=(1, window_height, window_width),
                resampling=rasterio.enums.Resampling.average,
                masked=True  # Use masked arrays to handle NaN values
            )
            
            # Ensure proper shape and squeeze out single-dimensional axes
            data = np.squeeze(data)
            
            # Convert masked array to regular array, replacing masked values with NaN
            if hasattr(data, 'mask'):
                data = data.filled(np.nan)
            
            # Debug information
            st.write(f"Data loaded with shape: {data.shape}, using downsample factor: {downsample_factor}")
            st.write(f"Memory usage after load: {get_memory_usage():.1f}MB")
            
            # Force garbage collection
            gc.collect()
            
            return data, src.bounds
            
    except Exception as e:
        st.error(f"Error in load_and_downsample_tiff: {str(e)}")
        # Force garbage collection on error
        gc.collect()
        raise

# Add this helper function to check memory before heavy operations
def check_memory_before_operation(operation_name, threshold_mb=1000):
    """Check if we have enough memory before a heavy operation"""
    current_memory = get_memory_usage()
    if current_memory > threshold_mb:
        st.warning(f"High memory usage ({current_memory:.1f}MB) before {operation_name}. Forcing garbage collection.")
        gc.collect()
        current_memory = get_memory_usage()
        if current_memory > threshold_mb:
            st.error(f"Insufficient memory for {operation_name}. Please try reducing the data size.")
            return False
    return True

def create_ridge_plot_optimized(values, title=None, max_lines=200):
    """Create ridge map with memory optimization"""
    try:
        # Handle NaN values and flip the array vertically
        values = np.flipud(values)
        values = np.nan_to_num(values, nan=np.nanmean(values))
        
        # Downsample data if too large
        if values.shape[0] > max_lines:
            step = values.shape[0] // max_lines
            values = values[::step]
        
        # Create figure with larger size for better quality
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create RidgeMap
        rm = RidgeMap()
        
        # Process and plot with improved parameters
        processed_values = rm.preprocess(
            values=values,
            lake_flatness=0.25,
            water_ntile=15,
            vertical_ratio=50
        )
        
        # Plot with basic parameters
        rm.plot_map(
            values=processed_values,
            label=title,
            ax=ax
        )
        
        # Clean up
        ax.axis('off')
        plt.close('all')
        gc.collect()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating ridge plot: {str(e)}")
        return None

# Add this helper function to format coordinates
def format_coordinates(lat, lon):
    """Format coordinates nicely with N/S and E/W indicators"""
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"({abs(lat):.2f}°{lat_dir}, {abs(lon):.2f}°{lon_dir})"

# Add this helper function at the top with the others
def downsample_for_3d(data, max_points=200000):
    """Downsample data for 3D plotting to prevent memory issues"""
    current_points = data.shape[0] * data.shape[1]
    if current_points > max_points:
        reduction_factor = int(np.sqrt(current_points / max_points))
        return data[::reduction_factor, ::reduction_factor]
    return data

def create_dem_plot(data, bounds, title=None):
    """Create DEM plot with memory optimization"""
    try:
        # Create figure with reasonable size
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create the plot with terrain colormap
        im = ax.imshow(data, 
                      cmap='terrain',
                      extent=[bounds.left, bounds.right, 
                             bounds.bottom, bounds.top])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Elevation (meters)')
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        # Format axis labels
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f°'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f°'))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Clean up
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating DEM plot: {str(e)}")
        return None

def create_3d_plot(data, bounds):
    """Create 3D terrain plot with memory optimization"""
    try:
        # Create coordinate meshgrid
        y = np.linspace(bounds.bottom, bounds.top, data.shape[0])
        x = np.linspace(bounds.left, bounds.right, data.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Create 3D surface plot
        fig = go.Figure(data=[
            go.Surface(
                z=data,
                x=X,
                y=Y,
                colorscale='earth',
                name='Elevation'
            )
        ])

        fig.update_layout(
            title='3D Terrain View',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=600,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D plot: {str(e)}")
        return None

# Process location data and select appropriate TIFF file(s)
if location_data:
    try:
        # Get TIFF path and bounds
        requested_bounds = location_data["bounds"]
        
        # Get TIFF file based on the view bounds center point
        center_lat = (requested_bounds[1] + requested_bounds[3]) / 2
        center_lon = (requested_bounds[0] + requested_bounds[2]) / 2
        
        # Check memory before loading TIFF
        if not check_memory_before_operation("loading TIFF file"):
            st.stop()
            
        tiff_path, downsample_factor = s3_handler.get_tiff_path(center_lat, center_lon)
            
        if not tiff_path:
            st.error("Could not find TIFF file for the given coordinates")
            st.stop()

        # Check memory before processing
        current_memory = get_memory_usage()
        st.sidebar.write(f"Current Memory Usage: {current_memory:.1f}MB")
        
        # Load and process the TIFF file with dynamic downsampling
        data, bounds = load_and_downsample_tiff(tiff_path, downsample_factor=downsample_factor)
        
        # Update progress
        progress_bar = st.progress(0)
        total_graphs = 6  # Total number of visualizations (3 original + 3 adjusted)
        graphs_completed = 0

        # Create visualizations with memory checks between each
        if data is not None:
            try:
                # Create two columns for the layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Region")
                    # Original DEM plot
                    if check_memory_before_operation("creating DEM plot"):
                        fig_dem = create_dem_plot(data, bounds, f"Full Region\n{format_coordinates(center_lat, center_lon)}")
                        if fig_dem:
                            st.pyplot(fig_dem)
                            plt.close(fig_dem)
                    graphs_completed += 1
                    progress_bar.progress(graphs_completed/total_graphs)
                    
                    # Check memory after each visualization
                    if not s3_handler.resource_manager.check_memory():
                        st.warning("High memory usage detected. Some visualizations may be skipped.")
                        gc.collect()
                    
                    # Ridge plot
                    if check_memory_before_operation("creating ridge plot"):
                        fig_ridge = create_ridge_plot_optimized(data, f"Full Region\n{format_coordinates(center_lat, center_lon)}")
                        if fig_ridge:
                            st.pyplot(fig_ridge)
                            plt.close(fig_ridge)
                    graphs_completed += 1
                    progress_bar.progress(graphs_completed/total_graphs)
                    
                    # Check memory again
                    if not s3_handler.resource_manager.check_memory():
                        st.warning("High memory usage detected. 3D visualization may be limited.")
                        gc.collect()
                    
                    # 3D surface plot
                    if check_memory_before_operation("creating 3D plot"):
                        plot_data = downsample_for_3d(data)
                        if s3_handler.resource_manager.check_memory():
                            fig_3d = create_3d_plot(plot_data, bounds)
                            if fig_3d:
                                st.plotly_chart(fig_3d)
                        graphs_completed += 1
                        progress_bar.progress(graphs_completed/total_graphs)
                    else:
                        st.warning("Insufficient memory for 3D visualization.")
                
                # Create adjusted view based on requested bounds
                with col2:
                    st.subheader("Selected Region")
                    try:
                        # Create bounding box and get adjusted image
                        adjusted_bounds = clamp_bounds(requested_bounds, bounds)
                        bbox = box(*adjusted_bounds)
                        
                        # Calculate center of adjusted region
                        adj_center_lat = (adjusted_bounds[3] + adjusted_bounds[1]) / 2
                        adj_center_lon = (adjusted_bounds[2] + adjusted_bounds[0]) / 2
                        
                        # Check memory before processing adjusted region
                        if not check_memory_before_operation("processing adjusted region"):
                            st.stop()
                            
                        # Re-open the TIFF file to use mask operation
                        with rasterio.open(tiff_path) as src:
                            # Get adjusted image using mask operation
                            out_image, out_transform = mask(src, [bbox], crop=True)
                            
                            if out_image.size > 0:
                                # Create a bounds object for the adjusted region
                                adjusted_bounds_obj = type('Bounds', (), {
                                    'left': adjusted_bounds[0],
                                    'right': adjusted_bounds[2],
                                    'bottom': adjusted_bounds[1],
                                    'top': adjusted_bounds[3]
                                })
                                
                                # Adjusted DEM plot
                                if check_memory_before_operation("creating adjusted DEM plot"):
                                    fig_adj_dem = create_dem_plot(out_image[0], 
                                                                adjusted_bounds_obj,
                                                                f"Selected Region\n{format_coordinates(adj_center_lat, adj_center_lon)}")
                                    if fig_adj_dem:
                                        st.pyplot(fig_adj_dem)
                                        plt.close(fig_adj_dem)
                                graphs_completed += 1
                                progress_bar.progress(graphs_completed/total_graphs)
                                
                                # Check memory
                                if not s3_handler.resource_manager.check_memory():
                                    st.warning("High memory usage detected. Some visualizations may be skipped.")
                                    gc.collect()
                                
                                # Adjusted ridge plot
                                if check_memory_before_operation("creating adjusted ridge plot"):
                                    fig_adj_ridge = create_ridge_plot_optimized(out_image[0], 
                                                                              f"Selected Region\n{format_coordinates(adj_center_lat, adj_center_lon)}")
                                    if fig_adj_ridge:
                                        st.pyplot(fig_adj_ridge)
                                        plt.close(fig_adj_ridge)
                                graphs_completed += 1
                                progress_bar.progress(graphs_completed/total_graphs)
                                
                                # Check memory
                                if not s3_handler.resource_manager.check_memory():
                                    st.warning("High memory usage detected. 3D visualization may be limited.")
                                    gc.collect()
                                
                                # Adjusted 3D plot
                                if check_memory_before_operation("creating adjusted 3D plot"):
                                    plot_data = downsample_for_3d(out_image[0])
                                    if s3_handler.resource_manager.check_memory():
                                        fig_adj_3d = create_3d_plot(plot_data, adjusted_bounds_obj)
                                        if fig_adj_3d:
                                            st.plotly_chart(fig_adj_3d)
                                    graphs_completed += 1
                                    progress_bar.progress(graphs_completed/total_graphs)
                                else:
                                    st.warning("Insufficient memory for 3D visualization.")
                            else:
                                st.error("Selected region is outside the available data range")
                                graphs_completed += 3  # Skip the remaining adjusted plots
                                progress_bar.progress(graphs_completed/total_graphs)
                                
                    except Exception as e:
                        st.error(f"Error creating adjusted views: {str(e)}")
                        st.write("Debug - Adjusted bounds:", adjusted_bounds)
                        st.write("Debug - Data shape:", data.shape if 'data' in locals() else "No data loaded")
                        graphs_completed += 3  # Skip the remaining adjusted plots
                        progress_bar.progress(graphs_completed/total_graphs)
                
                # Final cleanup
                gc.collect()
                
            except Exception as e:
                st.error(f"Error processing visualizations: {str(e)}")
        
        # Clear progress bar when done
        progress_bar.empty()

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("Debug - Error details:", str(e))
        gc.collect()

