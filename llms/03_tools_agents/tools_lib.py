"""Session 1's tools, saved as a module (tools practical, llms/03_tools_agents).

Imported by session 2's notebook and by server.py, the MCP server any client can start.
get_weather uses Open-Meteo (no key); web_search uses Tavily (TAVILY_API_KEY in the repository's .env).
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")      # the repository's .env; keys are never printed

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def get_weather(city: str, date: str | None = None, hour: int = 9) -> dict:
    """Weather in a city: now, or the forecast for a date (YYYY-MM-DD) at an hour (local time, 0-23).
    Includes low cloud, visibility and PM2.5, which decide whether distant mountains are visible."""

    # 1. city name -> coordinates
    found = requests.get(GEOCODING_URL, params={"name": city, "count": 1}, timeout=20).json()
    if not found.get("results"):
        return {"error": f"no city called {city!r}"}
    place = found["results"][0]
    where = {"latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto"}

    # 2. coordinates -> weather (now + hourly for 16 days) and air quality
    weather = requests.get(FORECAST_URL, timeout=20, params={
        **where,
        "forecast_days": 16,
        "current": "temperature_2m,precipitation,cloud_cover_low,visibility,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability,cloud_cover_low,visibility,wind_speed_10m",
    }).json()
    air = requests.get(AIR_QUALITY_URL, timeout=20, params={
        **where,
        "forecast_days": 5,
        "current": "pm2_5",
        "hourly": "pm2_5",
    }).json()

    result = {"city": place["name"], "country": place.get("country")}

    # 3a. no date (None or ""): the reading right now
    if not date:
        now = weather["current"]
        return {
            **result,
            "when": now["time"],
            "temperature_c": now["temperature_2m"],
            "low_cloud_pct": now["cloud_cover_low"],
            "visibility_km": round(now["visibility"] / 1000, 1),
            "precipitation_mm": now["precipitation"],
            "pm2_5": air.get("current", {}).get("pm2_5"),
        }

    # 3b. a date: look up that hour in the hourly forecast
    when = f"{date}T{int(hour):02d}:00"
    hourly = weather["hourly"]
    if when not in hourly["time"]:
        return {"error": f"no forecast for {when}: forecasts cover today and the next 15 days"}
    i = hourly["time"].index(when)

    air_hourly = air.get("hourly", {})
    air_times = air_hourly.get("time", [])
    pm2_5 = air_hourly["pm2_5"][air_times.index(when)] if when in air_times else None   # only 5 days ahead

    return {
        **result,
        "when": when,
        "temperature_c": hourly["temperature_2m"][i],
        "low_cloud_pct": hourly["cloud_cover_low"][i],
        "visibility_km": round(hourly["visibility"][i] / 1000, 1),
        "rain_chance_pct": hourly["precipitation_probability"][i],
        "pm2_5": pm2_5,
    }


def web_search(query: str) -> list[dict]:
    """Search the web for current facts, news or schedules. Returns the top 3 pages, each with a short text."""
    from tavily import TavilyClient
    response = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=3)
    return [
        {"title": page["title"], "url": page["url"], "text": page["content"][:500]}
        for page in response["results"]
    ]
