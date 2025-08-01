"""
YSU Scraper Demo - Functional Programming Version

This demo shows how to use the YSU scraper functions without actually making web requests.
"""

from ysu_scraper import (
    clean_name, setup_directories, setup_logging, 
    extract_year_of_birth, process_all_data
)
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import tempfile


def demo_clean_name():
    """Demonstrate name cleaning functionality."""
    print("=== Name Cleaning Demo ===")
    test_names = [
        "  John\nDoe  ",
        "Normal Name",
        "Multi\n\nLine  Name",
        "",
        "Արմեն Սարգսյան"
    ]
    
    for name in test_names:
        cleaned = clean_name(name)
        print(f"'{name}' -> '{cleaned}'")
    print()


def demo_extract_year():
    """Demonstrate birth year extraction."""
    print("=== Birth Year Extraction Demo ===")
    
    # Armenian HTML sample
    html = """
    <div class="personal-information">
        <div class="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above">
            Ծննդյան տարեթիվ
            1985
        </div>
    </div>
    """
    
    soup = BeautifulSoup(html, "html.parser")
    year = extract_year_of_birth(soup)
    print(f"Extracted year: {year}")
    
    # Test with missing data
    empty_html = "<html><body></body></html>"
    soup = BeautifulSoup(empty_html, "html.parser")
    year = extract_year_of_birth(soup)
    print(f"Missing data year: {year}")
    print()


def demo_data_processing():
    """Demonstrate data processing with sample data."""
    print("=== Data Processing Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Setup directories
        setup_directories(temp_path)
        print(f"Created directories in: {temp_path}")
        
        # Sample staff data
        staff_list = [
            {"name": "Արմեն Սարգսյան", "position": "Պրոֆեսոր", "url": "/staff/1", "img": "/img1.jpg"},
            {"name": "Անի Գրիգորյան", "position": "Դոցենտ", "url": "/staff/2", "img": "/img2.jpg"},
            {"name": "Դավիթ Աշոտյան", "position": "Ասիստենտ", "url": "/staff/3", "img": "/img3.jpg"}
        ]
        
        # Create sample HTML files
        htmls_dir = temp_path / "htmls"
        
        # File with birth year
        html_with_year = """
        <div class="personal-information">
            <div class="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above">
                Ծննդյան տարեթիվ
                1985
            </div>
        </div>
        """
        (htmls_dir / "00_Արմեն Սարգսյան.html").write_text(html_with_year, encoding="utf-8")
        
        # File without birth year
        (htmls_dir / "01_Անի Գրիգորյան.html").write_text("<html></html>", encoding="utf-8")
        
        # File with different birth year
        html_with_year2 = """
        <div class="personal-information">
            <div class="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above">
                Ծննդյան տարեթիվ
                1990
            </div>
        </div>
        """
        (htmls_dir / "02_Դավիթ Աշոտյան.html").write_text(html_with_year2, encoding="utf-8")
        
        # Process the data
        df = process_all_data(staff_list, temp_path)
        
        print("Processed data:")
        print(df[['name', 'position', 'year']].to_string(index=False))
        
        # Show file statistics
        csv_files = list(temp_path.glob("ysu_staff_data_*.csv"))
        if csv_files:
            print(f"\nSaved data to: {csv_files[0].name}")
    print()


def demo_logging():
    """Demonstrate logging setup."""
    print("=== Logging Demo ===")
    
    # Setup logging
    setup_logging("INFO")
    print("Logging configured - check ysu_scraper.log file")
    print()


def main():
    """Run all demos."""
    print("YSU Scraper Functional Programming Demo")
    print("=" * 40)
    
    demo_clean_name()
    demo_extract_year()
    demo_logging()
    demo_data_processing()
    
    print("Demo completed! Check the generated log file and CSV output.")


if __name__ == "__main__":
    main()
