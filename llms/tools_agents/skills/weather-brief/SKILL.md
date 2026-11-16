---
name: weather-brief
description: The required format for any weather brief or short weather report. Load it BEFORE calling get_weather whenever the user says 'weather brief'.
---
# Weather brief

Get the numbers with `get_weather`; never guess them. Then answer in Armenian, in exactly this format, one line
each, and nothing else:

Եղանակ: <city>, <date> <hour>:00
Ջերմաստիճան: <temperature_c> °C
Ցածր ամպեր: <low_cloud_pct>%
Տեսանելիություն: <visibility_km> կմ
Արարատը: երևում է / չի երևում

Rule of thumb for the last line (a house rule, not meteorology): Ararat counts as visible when visibility is at
least 20 km and low cloud is at most 30%.
