import math

def calculate_zoom_level(bounds):
    """
    Calculate the appropriate zoom level to fit the given bounds.
    
    Args:
        bounds (dict): Dictionary containing north, south, east, west coordinates
        
    Returns:
        int: Zoom level (0-18, where 0 is most zoomed out)
    """
    # Get the bounds
    north = bounds['bounds']['north']
    south = bounds['bounds']['south']
    east = bounds['bounds']['east']
    west = bounds['bounds']['west']
    
    # Calculate the angular distance
    lat_diff = abs(north - south)
    lon_diff = abs(east - west)
    
    # Use the larger of the two differences to determine zoom
    max_diff = max(lat_diff, lon_diff)
    
    # Zoom level formula based on angular distance
    # This is an approximation that works well for most cases
    # 360 degrees = zoom 0
    # 180 degrees = zoom 1
    # 90 degrees = zoom 2
    # etc.
    zoom = round(math.log2(360 / max_diff))
    
    # Clamp zoom level between 0 and 18 (common min/max zoom levels)
    zoom = min(max(zoom, 0), 18)
    
    # Adjust zoom to show slightly more context
    zoom = max(zoom - 1, 0)
    
    return zoom 