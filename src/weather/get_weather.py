import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_weather_data(lat, lon):
    """
    Get current weather and alerts for a location
    """
    # Get API key from environment variable
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    # API endpoint for current weather and alerts
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&units=imperial&appid={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant information
        current = data.get('current', {})
        alerts = data.get('alerts', [])
        
        weather_info = {
            'temperature': current.get('temp'),
            'feels_like': current.get('feels_like'),
            'humidity': current.get('humidity'),
            'wind_speed': current.get('wind_speed'),
            'description': current.get('weather', [{}])[0].get('description', '').title(),
            'alerts': [
                {
                    'event': alert.get('event'),
                    'description': alert.get('description'),
                    'start': alert.get('start'),
                    'end': alert.get('end')
                }
                for alert in alerts
            ]
        }
        
        return weather_info
        
    except Exception as e:
        return {
            'error': f"Failed to get weather data: {str(e)}"
        } 