import json

import requests
from aipe.llm import init_payx_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

model = init_payx_chat_model(model="gpt-41", model_provider="azure_openai")


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """Get current temperature in Fahrenheit and weather code for given coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        JSON string with temperature_fahrenheit and weather_code (do not include the code in your response, translate it to plain English)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "fahrenheit",
    }

    weather = requests.get(url, params=params).json()["current"]
    temperature = weather["temperature_2m"]
    weather_code = weather["weather_code"]
    result = {"temperature_fahrenheit": temperature, "weather_code": weather_code}

    return json.dumps(result)


@tool
def get_user_preferences(user_id: str) -> str:
    """Get a user's saved preferences."""
    # Simulate a user database
    preferences = {
        "alice": "Loves sci-fi movies, prefers warm weather destinations",
        "bob": "Enjoys comedy films, likes cold climates for travel",
    }
    return preferences.get(user_id.lower(), "No preferences found")


@tool
def book_recommendation(genre: str, user_preferences: str = "") -> str:
    """Get personalized movie recommendations based on genre and user preferences."""
    recommendations = {
        "sci-fi": "Based on your preferences, try: Arrival, Ex Machina, or The Martian",
        "comedy": "Based on your preferences, try: The Big Lebowski, Anchorman, or Bridesmaids",
    }
    return recommendations.get(genre.lower(), "No recommendations available")


# Create a helpful assistant agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_user_preferences, book_recommendation],
    system_prompt="""You are a helpful personal assistant. 
    
    You can:
    - Check weather for any city
    - Look up user preferences
    - Recommend movies based on preferences
    
    Always be friendly and personalize your responses based on user preferences.""",
)
