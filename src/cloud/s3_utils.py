import os
from pathlib import Path
import streamlit as st
import time
import dotenv
import psutil
import threading
from typing import Optional
import rasterio
from rasterio.io import MemoryFile
import numpy as np

class ResourceManager:
    def __init__(self, memory_limit_mb: Optional[float] = None):
        """
        Initialize resource manager with optional memory limit.
        If no limit provided, uses 75% of system memory.
        """
        self.system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        self.memory_limit = memory_limit_mb or self.system_memory * 0.75
        self.lock = threading.Lock()
        self._cleanup_threshold = self.memory_limit * 0.9  # 90% of limit
        
    def get_available_memory(self) -> float:
        """Get available memory in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def get_used_memory(self) -> float:
        """Get current process memory usage in MB"""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    def check_memory(self) -> bool:
        """
        Check if memory usage is within limits.
        Returns True if memory is OK, False if we're over limit.
        """
        with self.lock:
            used = self.get_used_memory()
            if used > self.memory_limit:
                return False
            return True
    
    def estimate_tiff_memory(self, file_size_mb: float, processing_factor: float = 2.5) -> float:
        """
        Estimate memory needed for TIFF processing.
        processing_factor accounts for temporary copies during processing.
        """
        return file_size_mb * processing_factor
    
    def can_process_file(self, file_size_mb: float) -> tuple[bool, float]:
        """
        Check if we can process a file of given size.
        Returns (can_process, recommended_downsample_factor)
        """
        estimated_memory = self.estimate_tiff_memory(file_size_mb)
        available_memory = self.get_available_memory()
        
        if estimated_memory < available_memory * 0.8:  # Leave 20% buffer
            return True, 1
        
        # Calculate downsample factor needed
        downsample_factor = int(np.ceil(np.sqrt(estimated_memory / (available_memory * 0.8))))
        return False, max(2, downsample_factor)  # Minimum downsample factor of 2

class S3Handler:
    def __init__(self, bucket_name, region_name):
        if not bucket_name:
            raise ValueError("bucket_name must be provided")
            
        self.bucket_name = bucket_name
        self.mount_point = Path("/home/ubuntu/s3bucket")
        
        # Verify mount point exists
        if not self.mount_point.exists():
            raise RuntimeError(f"Mount point {self.mount_point} does not exist")
            
        # Initialize resource manager
        self.resource_manager = ResourceManager()
        
    def get_tiff_path(self, lat, lon):
        """
        Get the local path to the TIFF file for given coordinates.
        Returns tuple of (path, recommended_downsample_factor)
        """
        tiff_key = self.get_tiff_key(lat, lon)
        if not tiff_key:
            return None, 1
            
        file_path = self.mount_point / tiff_key
        
        # Check if file exists
        if not file_path.exists():
            st.error(f"File not found at {file_path}")
            return None, 1
            
        # Get file size and check if we need downsampling
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        _, downsample = self.resource_manager.can_process_file(file_size_mb)
        
        if downsample > 1:
            st.warning(f"Large file detected ({file_size_mb:.1f}MB). Using downsampling factor {downsample}x")
            
        return str(file_path), downsample
        
    def get_tiff_key(self, lat, lon):
        """
        Get the relative path to the TIFF file covering the given coordinates.
        For example: (40.30, -74.00) should return '10_DEM_y40x-80.tif'
        """
        # Check if coordinates are in range
        if not (-180 <= lon <= -10 and 10 <= lat <= 80):
            raise ValueError(f"Coordinates ({lat}, {lon}) are outside available range. "
                          "Available range: longitude (-180 to -10), latitude (10 to 80)")
        
        # Floor to nearest 10 for latitude
        lat_base = int(lat // 10) * 10
        
        # For negative longitude, get the correct 10-degree tile
        # e.g., -74 should be in file x-80 (covers -80 to -70)
        lon_base = int(lon // 10) * 10
        if lon_base > lon:  # If we rounded up, go to next lower tile
            lon_base -= 10
        
        # Construct filename
        filename = f"10_DEM_y{lat_base}x-{abs(lon_base)}.tif"
        
        # Show debug info
        print(f"Input coordinates: ({lat}, {lon})")
        print(f"Latitude base: {lat_base}")
        print(f"Longitude base: {lon_base}")
        print(f"Selected file: {filename}")
        
        return filename

    def clamp_bounds(self, requested_bounds, tiff_bounds):
        """
        Clamp requested bounds to stay within TIFF file bounds.
        Returns tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        min_lon = max(min(requested_bounds[0], requested_bounds[2]), tiff_bounds.left)
        max_lon = min(max(requested_bounds[0], requested_bounds[2]), tiff_bounds.right)
        min_lat = max(min(requested_bounds[1], requested_bounds[3]), tiff_bounds.bottom)
        max_lat = min(max(requested_bounds[1], requested_bounds[3]), tiff_bounds.top)
        
        return (min_lon, min_lat, max_lon, max_lat)

class S3MountHandler:
    mount_point = os.getenv('MOUNT_POINT')
    def __init__(self, mount_point=mount_point):
        self.mount_point = Path(mount_point)
        
        # Detailed mount point verification
        st.write("Checking S3 mount status:")
        if not self.mount_point.exists():
            raise RuntimeError(f"Mount point {mount_point} does not exist")
        
        # Check if directory is actually mounted
        try:
            st.write(f"Mount point contents:")
            st.write(f"- Is directory: {self.mount_point.is_dir()}")
            st.write(f"- Is mount: {os.path.ismount(self.mount_point)}")
            
            # List all contents recursively
            st.write("Searching for TIFF files...")
            all_files = list(self.mount_point.rglob('*.tif'))
            if all_files:
                st.write(f"Found {len(all_files)} TIFF files:")
                for f in all_files:
                    st.write(f"- {f.relative_to(self.mount_point)}")
            else:
                st.error("No TIFF files found in mount point or subdirectories")
                
            # Check mount point stats
            stats = os.statvfs(self.mount_point)
            total_space = stats.f_blocks * stats.f_frsize
            st.write(f"Mount point total space: {total_space / (1024**3):.2f} GB")
            
        except Exception as e:
            st.error(f"Error checking mount point: {str(e)}")
            raise RuntimeError(f"Failed to verify S3 mount: {str(e)}")

    def get_tiff_key(self, lat, lon):
        """
        Select appropriate TIFF file based on coordinates.
        Uses the minimum bounds (bottom-left corner) to select file.
        """
        # Check if coordinates are in range
        if not (-180 <= lon <= -10 and 10 <= lat <= 80):
            raise ValueError(f"Coordinates ({lat}, {lon}) are outside available range. "
                          "Available range: longitude (-180 to -10), latitude (10 to 80)")
        
        # Floor to nearest 10 for latitude
        lat_base = int(lat // 10) * 10
        
        # For negative longitude, round down to next lower 10
        # e.g., -74 should use -70 as base (file covers -80 to -70)
        if lon < 0:
            lon_base = int((lon + 10) // 10) * 10  # Add 10 to get correct base for negative values
        else:
            lon_base = int(lon // 10) * 10
        
        # Always use negative format for longitude in filename
        filename = f"10_DEM_y{lat_base}x-{abs(lon_base)}.tif"
        
        # Show the actual coverage area
        st.write(f"Input coordinates: ({lat:.2f}, {lon:.2f})")
        st.write(f"Selected TIFF file: {filename}")
        st.write(f"File coverage:")
        st.write(f"Latitude: {lat_base}° to {lat_base + 10}°N")
        st.write(f"Longitude: {lon_base - 10}° to {lon_base}°W")  # Fixed longitude range display
        
        file_path = self.mount_point / filename
        
        # More detailed error reporting
        if not self.mount_point.exists():
            st.error(f"Mount point {self.mount_point} does not exist")
            return None
            
        st.write(f"Checking path: {file_path}")
        if file_path.exists():
            st.write("File found!")
            return filename
        else:
            st.error(f"File not found at {file_path}")
            # List files in mount point with similar pattern
            similar_files = list(self.mount_point.glob('10_DEM_*.tif'))
            if similar_files:
                st.write("Available files with similar pattern:")
                for f in similar_files:
                    st.write(f"- {f.name}")
            else:
                st.error("No TIFF files found in mount point")
            return None

    def get_bounds(self, filename):
        """
        Get the bounds for a given TIFF file based on its name.
        Returns (left, bottom, right, top) - same order as rasterio bounds
        """
        # Parse lat/lon from filename
        parts = filename.split('_')
        lat_part = parts[2][1:]  # Remove 'y'
        lon_part = parts[3][1:-4]  # Remove 'x' and '.tif'
        
        lat = int(lat_part)
        lon = -int(lon_part)  # Convert back to negative
        
        # Return bounds in rasterio order (left, bottom, right, top)
        return (lon, lat, lon + 10, lat + 10)

    def get_tiff_path(self, lat, lon):
        """Get the full path to the TIFF file for given coordinates."""
        tiff_key = self.get_tiff_key(lat, lon)
        if not tiff_key:
            # Try searching in subdirectories
            st.write("Searching for file in subdirectories...")
            try:
                matches = list(self.mount_point.rglob(tiff_key))
                if matches:
                    st.write(f"Found file in: {matches[0].relative_to(self.mount_point)}")
                    return str(matches[0])
            except Exception as e:
                st.error(f"Error searching subdirectories: {str(e)}")
            return None
            
        return str(self.mount_point / tiff_key)

    def clamp_bounds(self, requested_bounds, tiff_bounds):
        """
        Clamp requested bounds to stay within TIFF file bounds.
        Returns tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        min_lon = max(min(requested_bounds[0], requested_bounds[2]), tiff_bounds.left)
        max_lon = min(max(requested_bounds[0], requested_bounds[2]), tiff_bounds.right)
        min_lat = max(min(requested_bounds[1], requested_bounds[3]), tiff_bounds.bottom)
        max_lat = min(max(requested_bounds[1], requested_bounds[3]), tiff_bounds.top)
        
        return (min_lon, min_lat, max_lon, max_lat) 