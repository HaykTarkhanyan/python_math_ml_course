def clean_name(name: str) -> str:
    # return " ".join(name.split())
    return name.replace("\n", "").replace("  ", " ").strip()
