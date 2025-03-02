import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / 'data' / 'urbexfun.db'

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def get_all_states():
    """Get all states from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT state_name FROM states ORDER BY state_name")
    states = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return states

def get_cities_in_state(state_name):
    """Get all cities in a given state"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.city_name 
        FROM cities c
        JOIN states s ON c.state_id = s.state_id
        WHERE s.state_name = ?
        ORDER BY c.city_name
    """, (state_name,))
    
    cities = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return cities

def get_city_info(city_name, state_name):
    """Get all information for a specific city"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            c.city_name,
            s.state_name,
            s.state_abbrev,
            c.population_2024,
            c.population_2020,
            c.annual_change,
            c.density_per_mile2,
            c.area_mile2,
            coord.latitude,
            coord.longitude,
            GROUP_CONCAT(z.zip_code) as zip_codes
        FROM cities c
        JOIN states s ON c.state_id = s.state_id
        LEFT JOIN lat_long_coords coord ON c.city_id = coord.city_id
        LEFT JOIN zip_codes z ON c.city_id = z.city_id
        WHERE c.city_name = ? AND s.state_name = ?
        GROUP BY c.city_id
    """, (city_name, state_name))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # Convert to dictionary and parse zip codes string
        data = dict(result)
        if data['zip_codes']:
            data['zip_codes'] = data['zip_codes'].split(',')
        else:
            data['zip_codes'] = []
        return data
    return None

def get_city_zipcodes(city_name, state_name):
    """Get all zip codes for a specific city"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT z.zip_code
        FROM zip_codes z
        JOIN cities c ON z.city_id = c.city_id
        JOIN states s ON c.state_id = s.state_id
        WHERE c.city_name = ? AND s.state_name = ?
        ORDER BY z.zip_code
    """, (city_name, state_name))
    
    zipcodes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return zipcodes

def get_city_ips(city_name, state_name):
    """Get all IP addresses for a specific city"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ip.ip_address
        FROM ip_addresses ip
        JOIN cities c ON ip.city_id = c.city_id
        JOIN states s ON c.state_id = s.state_id
        WHERE c.city_name = ? AND s.state_name = ?
        ORDER BY ip.ip_address
    """, (city_name, state_name))
    
    ip_addresses = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ip_addresses 