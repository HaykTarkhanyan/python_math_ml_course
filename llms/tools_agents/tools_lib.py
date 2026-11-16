"""Session 1's tools, saved as a module (tools practical, llms/tools_agents).

Imported by session 2's notebook and by server.py, the MCP server any client can start.
get_weather uses Open-Meteo (no key); web_search uses Tavily (TAVILY_API_KEY in the repository's .env).
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")      # the repository's .env; keys are never printed


def get_weather(city: str, date: str | None = None, hour: int = 9) -> dict:
    """Weather in a city: now, or the forecast for a date (YYYY-MM-DD) at an hour (local time, 0-23).
    Includes low cloud, visibility and PM2.5, which decide whether distant mountains are visible."""
    g = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1},
                     timeout=20).json()
    if not g.get("results"):
        return {"error": f"no city called {city!r}"}
    p = g["results"][0]
    loc = {"latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto"}
    w = requests.get("https://api.open-meteo.com/v1/forecast", timeout=20, params={**loc, "forecast_days": 16,
        "current": "temperature_2m,precipitation,cloud_cover_low,visibility,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability,cloud_cover_low,visibility,wind_speed_10m"}).json()
    a = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=20,
                     params={**loc, "current": "pm2_5", "hourly": "pm2_5", "forecast_days": 5}).json()
    out = {"city": p["name"], "country": p.get("country")}
    if date is None:
        c = w["current"]
        return {**out, "when": c["time"], "temperature_c": c["temperature_2m"], "low_cloud_pct": c["cloud_cover_low"],
                "visibility_km": round(c["visibility"] / 1000, 1), "precipitation_mm": c["precipitation"],
                "pm2_5": a.get("current", {}).get("pm2_5")}
    t, h = f"{date}T{int(hour):02d}:00", w["hourly"]
    if t not in h["time"]:
        return {"error": f"no forecast for {t}: forecasts reach 16 days ahead"}
    i, ah = h["time"].index(t), a.get("hourly", {})
    return {**out, "when": t, "temperature_c": h["temperature_2m"][i], "low_cloud_pct": h["cloud_cover_low"][i],
            "visibility_km": round(h["visibility"][i] / 1000, 1), "rain_chance_pct": h["precipitation_probability"][i],
            "pm2_5": ah["pm2_5"][ah["time"].index(t)] if t in ah.get("time", []) else None}


def web_search(query: str) -> list[dict]:
    """Search the web for current facts, news or schedules. Returns the top 3 pages, each with a short text."""
    from tavily import TavilyClient
    r = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=3)
    return [{"title": x["title"], "url": x["url"], "text": x["content"][:500]} for x in r["results"]]
