"""
YSU Faculty Staff Scraper

A minimalistic web scraper for YSU faculty staff information.
Includes logging, error handling, and parallel processing capabilities.
"""

import logging
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from joblib import Parallel, delayed
from datetime import datetime


# Global configuration
BASE_URL = "https://www.ysu.am"
SAVE_DIR = Path("data")
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ysu_scraper.log'),
            logging.StreamHandler()
        ]
    )
    logger.info("YSU Scraper initialized")


def setup_directories(save_dir: Path = SAVE_DIR) -> None:
    """Create necessary directories."""
    save_dir.mkdir(exist_ok=True)
    (save_dir / "htmls").mkdir(exist_ok=True)
    logger.info(f"Created directories: {save_dir}")


def clean_name(name: str) -> str:
    """Clean and normalize names."""
    return name.strip().replace("\n", " ").replace("  ", " ")


def get_staff_list(faculty_url: str) -> List[Dict[str, str]]:
    """
    Scrape the main faculty staff listing page.
    
    Args:
        faculty_url: URL of the faculty staff page
        
    Returns:
        List of dictionaries containing staff information
    """
    logger.info(f"Scraping staff list from: {faculty_url}")
    
    try:
        response = requests.get(faculty_url, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully fetched main page (status: {response.status_code})")
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch main page: {e}")
        raise
        
    soup = BeautifulSoup(response.content, "html.parser")
    grid = soup.find("div", class_="view-content row")
    
    if not grid:
        logger.error("Could not find staff grid on page")
        raise ValueError("Staff grid not found")
        
    cells = grid.find_all("div", class_="col-6 col-md-3 views-row")
    logger.info(f"Found {len(cells)} staff members")
    
    staff_list = []
    
    for cell in cells:
        try:
            url = cell.find("a")["href"]
            img = cell.find("img")["src"]
            name = clean_name(cell.find("div", class_="user-full-name").text)
            position_elem = cell.find("div", class_="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item")
            position = position_elem.text.strip() if position_elem else "Unknown"
            
            staff_list.append({
                "name": name,
                "position": position,
                "url": url,
                "img": img
            })
            
        except Exception as e:
            logger.warning(f"Error processing staff member: {e}")
            continue
            
    logger.info(f"Successfully extracted {len(staff_list)} staff members")
    return staff_list


def download_staff_page(index: int, staff_info: Dict[str, str], save_dir: Path = SAVE_DIR) -> Optional[Path]:
    """
    Download individual staff member's page.
    
    Args:
        index: Staff member index
        staff_info: Dictionary with staff information
        save_dir: Directory to save files
        
    Returns:
        Path to saved file or None if failed
    """
    name = clean_name(staff_info["name"])
    filename = save_dir / "htmls" / f"{index:02d}_{name}.html"
    
    if filename.exists():
        logger.debug(f"File {filename} already exists. Skipping.")
        return filename
        
    url = BASE_URL + staff_info["url"]
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(filename, "w", encoding="utf-8") as file:
            file.write(response.text)
            
        logger.debug(f"Downloaded: {name} -> {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        return None


def download_all_pages(staff_list: List[Dict[str, str]], n_jobs: int = 4, 
                      save_dir: Path = SAVE_DIR) -> List[Optional[Path]]:
    """
    Download all staff pages in parallel.
    
    Args:
        staff_list: List of staff information
        n_jobs: Number of parallel jobs
        save_dir: Directory to save files
        
    Returns:
        List of file paths
    """
    logger.info(f"Downloading {len(staff_list)} pages with {n_jobs} workers")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(download_staff_page)(i, staff_info, save_dir) 
        for i, staff_info in enumerate(staff_list)
    )
    
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Downloaded {successful}/{len(staff_list)} pages successfully")
    
    return results


def extract_year_of_birth(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract year of birth from individual staff page.
    
    Args:
        soup: BeautifulSoup object of staff page
        
    Returns:
        Year of birth or None if not found
    """
    try:
        personal_info = soup.find("div", class_="personal-information")
        if not personal_info:
            return None
            
        date_of_birth = personal_info.find("div", class_="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above")
        if not date_of_birth:
            return None
            
        date_text_list = date_of_birth.text.strip().split("\n")
        if len(date_text_list) < 2:
            return None
            
        title = date_text_list[0].strip()
        year = date_text_list[1].strip()
        
        if title == "Ծննդյան տարեթիվ":
            return year
        else:
            logger.warning(f"Unexpected birth year title: {title}")
            return None
            
    except Exception as e:
        logger.debug(f"Error extracting birth year: {e}")
        return None


def process_all_data(staff_list: List[Dict[str, str]], save_dir: Path = SAVE_DIR) -> pd.DataFrame:
    """
    Process all downloaded data and extract birth years.
    
    Args:
        staff_list: List of staff information
        save_dir: Directory containing downloaded files
        
    Returns:
        DataFrame with all processed data
    """
    logger.info("Processing all staff data...")
    
    df = pd.DataFrame(staff_list)
    df['year'] = None
    
    for i, row in df.iterrows():
        filename = save_dir / "htmls" / f"{i:02d}_{clean_name(row['name'])}.html"
        
        try:
            if not filename.exists():
                logger.warning(f"File not found: {filename}")
                continue
                
            with open(filename, "r", encoding="utf-8") as f:
                html = f.read()
                
            soup = BeautifulSoup(html, "html.parser")
            year = extract_year_of_birth(soup)
            df.loc[i, "year"] = year
            
        except Exception as e:
            logger.error(f"Error processing {row['name']}: {e}")
            continue
            
    # Save processed data
    output_file = save_dir / f"ysu_staff_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"Saved processed data to: {output_file}")
    
    return df


def run_full_scrape(faculty_url: str, n_jobs: int = 4, save_dir: Path = SAVE_DIR,
                   log_level: str = "INFO") -> pd.DataFrame:
    """
    Run the complete scraping process.
    
    Args:
        faculty_url: URL of the faculty staff page
        n_jobs: Number of parallel jobs for downloading
        save_dir: Directory to save files
        log_level: Logging level
        
    Returns:
        DataFrame with all processed data
    """
    # Setup
    setup_logging(log_level)
    setup_directories(save_dir)
    
    logger.info("Starting full scrape process")
    
    # Step 1: Get staff list
    staff_list = get_staff_list(faculty_url)
    
    # Step 2: Download all pages
    download_all_pages(staff_list, n_jobs=n_jobs, save_dir=save_dir)
    
    # Step 3: Process all data
    df = process_all_data(staff_list, save_dir=save_dir)
    
    logger.info("Full scrape process completed")
    return df


def main():
    """Main function to run the scraper."""
    setup_logging("INFO")
    setup_directories(SAVE_DIR)
    
    faculty_url = "https://www.ysu.am/faculty/516/staff"
    
    try:
        df = run_full_scrape(faculty_url, n_jobs=4, save_dir=SAVE_DIR)
        print(f"\nSuccessfully scraped {len(df)} staff members:")
        print(df[['name', 'position', 'year']].head())
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    main()
