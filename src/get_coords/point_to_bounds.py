import math

def get_bounds_from_point(center_lat, center_lon, area_sq_miles):
    """
    Calculate bounding box from a center point and area.
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        area_sq_miles (float): Area in square miles
    
    Returns:
        dict: Bounding box coordinates and calculated distances
    """
    # Calculate side length from area (assuming square)
    side_length = math.sqrt(area_sq_miles)
    
    # Convert half of side length to degrees
    # 1 degree of latitude = ~69 miles
    # 1 degree of longitude = ~69 miles * cos(latitude)
    lat_offset = (side_length/2) / 69.0
    lon_offset = (side_length/2) / (69.0 * math.cos(math.radians(abs(center_lat))))
    
    bounds = {
        'center': {
            'lat': center_lat,
            'lon': center_lon
        },
        'bounds': {
            'north': center_lat + lat_offset,
            'south': center_lat - lat_offset,
            'east': center_lon + lon_offset,
            'west': center_lon - lon_offset
        },
        'distances': {
            'side_length_miles': side_length,
            'lat_offset_miles': lat_offset * 69,
            'lon_offset_miles': lon_offset * 69 * math.cos(math.radians(center_lat))
        }
    }
    
    return bounds

def print_bounds_info(bounds):
    """Print formatted bounds information"""
    print("\nCenter Point:")
    print(f"Latitude: {bounds['center']['lat']:.4f}")
    print(f"Longitude: {bounds['center']['lon']:.4f}")
    
    print("\nBounding Box:")
    print(f"North: {bounds['bounds']['north']:.4f}")
    print(f"South: {bounds['bounds']['south']:.4f}")
    print(f"East: {bounds['bounds']['east']:.4f}")
    print(f"West: {bounds['bounds']['west']:.4f}")
    
    print("\nCalculated Distances:")
    print(f"Side Length: {bounds['distances']['side_length_miles']:.2f} miles")
    print(f"Latitude Offset: {bounds['distances']['lat_offset_miles']:.2f} miles")
    print(f"Longitude Offset: {bounds['distances']['lon_offset_miles']:.2f} miles")

if __name__ == "__main__":
    # Example usage
    # Test with New York City's coordinates and area
    center_lat = 40.7306
    center_lon = -73.9866
    area = 300.46  # NYC area in square miles
    
    print("Testing with New York City:")
    bounds = get_bounds_from_point(center_lat, center_lon, area)
    print_bounds_info(bounds)
    
    # Test with a smaller city
    print("\nTesting with a smaller area (50 sq miles):")
    bounds = get_bounds_from_point(center_lat, center_lon, 50)
    print_bounds_info(bounds)
    
    # Test with a larger city
    print("\nTesting with a larger area (1000 sq miles):")
    bounds = get_bounds_from_point(center_lat, center_lon, 1000)
    print_bounds_info(bounds) 