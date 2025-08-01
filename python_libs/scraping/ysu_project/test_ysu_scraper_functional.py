"""
Tests for YSU Faculty Staff Scraper (Functional Version)

Run with: pytest test_ysu_scraper_functional.py -v
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup

from ysu_scraper import (
    clean_name, get_staff_list, download_staff_page, download_all_pages,
    extract_year_of_birth, process_all_data, run_full_scrape, setup_directories
)


class TestYSUScraperFunctions:
    """Test cases for functional YSU Scraper."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing."""
        return """
        <div class="view-content row">
            <div class="col-6 col-md-3 views-row">
                <a href="/staff/john-doe">
                    <img src="/images/john.jpg">
                </a>
                <div class="user-full-name">John Doe</div>
                <div class="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item">
                    Professor
                </div>
            </div>
            <div class="col-6 col-md-3 views-row">
                <a href="/staff/jane-smith">
                    <img src="/images/jane.jpg">
                </a>
                <div class="user-full-name">Jane Smith</div>
                <div class="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item">
                    Associate Professor
                </div>
            </div>
        </div>
        """
    
    @pytest.fixture
    def sample_staff_page(self):
        """Sample individual staff page HTML."""
        return """
        <div class="personal-information">
            <div class="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above">
                Ծննդյան տարեթիվ
                1985
            </div>
        </div>
        """
    
    def test_clean_name(self):
        """Test name cleaning functionality."""
        assert clean_name("  John\nDoe  ") == "John Doe"
        assert clean_name("Normal Name") == "Normal Name"
        assert clean_name("") == ""
        assert clean_name("Multi\n\nLine  Name") == "Multi Line Name"
    
    def test_setup_directories(self, temp_dir):
        """Test directory setup."""
        setup_directories(temp_dir)
        assert temp_dir.exists()
        assert (temp_dir / "htmls").exists()
    
    @patch('ysu_scraper.requests.get')
    def test_get_staff_list_success(self, mock_get, sample_html):
        """Test successful staff list retrieval."""
        mock_response = Mock()
        mock_response.content = sample_html.encode()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        staff_list = get_staff_list("http://example.com")
        
        assert len(staff_list) == 2
        assert staff_list[0]["name"] == "John Doe"
        assert staff_list[0]["position"] == "Professor"
        assert staff_list[0]["url"] == "/staff/john-doe"
        assert staff_list[1]["name"] == "Jane Smith"
        assert staff_list[1]["position"] == "Associate Professor"
    
    @patch('ysu_scraper.requests.get')
    def test_get_staff_list_network_error(self, mock_get):
        """Test network error handling."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            get_staff_list("http://example.com")
    
    @patch('ysu_scraper.requests.get')
    def test_get_staff_list_empty_grid(self, mock_get):
        """Test handling of empty staff grid."""
        mock_response = Mock()
        mock_response.content = "<html><body></body></html>".encode()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Staff grid not found"):
            get_staff_list("http://example.com")
    
    @patch('ysu_scraper.requests.get')
    def test_download_staff_page_success(self, mock_get, temp_dir):
        """Test successful page download."""
        setup_directories(temp_dir)
        
        mock_response = Mock()
        mock_response.text = "<html>Test content</html>"
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        staff_info = {"name": "John Doe", "url": "/staff/john-doe"}
        result = download_staff_page(0, staff_info, temp_dir)
        
        assert result is not None
        assert result.exists()
        assert "00_John Doe.html" in str(result)
        
        # Check file content
        content = result.read_text(encoding="utf-8")
        assert content == "<html>Test content</html>"
    
    @patch('ysu_scraper.requests.get')
    def test_download_staff_page_failure(self, mock_get, temp_dir):
        """Test download failure handling."""
        setup_directories(temp_dir)
        mock_get.side_effect = Exception("Download failed")
        
        staff_info = {"name": "John Doe", "url": "/staff/john-doe"}
        result = download_staff_page(0, staff_info, temp_dir)
        
        assert result is None
    
    def test_download_staff_page_existing_file(self, temp_dir):
        """Test skipping existing files."""
        setup_directories(temp_dir)
        
        # Create existing file
        existing_file = temp_dir / "htmls" / "00_John Doe.html"
        existing_file.write_text("existing content", encoding="utf-8")
        
        staff_info = {"name": "John Doe", "url": "/staff/john-doe"}
        result = download_staff_page(0, staff_info, temp_dir)
        
        assert result == existing_file
        assert result.read_text(encoding="utf-8") == "existing content"
    
    def test_extract_year_of_birth_success(self, sample_staff_page):
        """Test successful birth year extraction."""
        soup = BeautifulSoup(sample_staff_page, "html.parser")
        year = extract_year_of_birth(soup)
        
        assert year == "1985"
    
    def test_extract_year_of_birth_not_found(self):
        """Test birth year extraction when not found."""
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        year = extract_year_of_birth(soup)
        
        assert year is None
    
    def test_extract_year_of_birth_wrong_title(self):
        """Test birth year extraction with wrong title."""
        html = """
        <div class="personal-information">
            <div class="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above">
                Wrong Title
                1985
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        year = extract_year_of_birth(soup)
        
        assert year is None
    
    def test_process_all_data(self, temp_dir, sample_staff_page):
        """Test data processing functionality."""
        setup_directories(temp_dir)
        
        # Create test files
        staff_list = [
            {"name": "John Doe", "position": "Professor", "url": "/john", "img": "/john.jpg"},
            {"name": "Jane Smith", "position": "Associate Professor", "url": "/jane", "img": "/jane.jpg"}
        ]
        
        # Create test HTML files
        htmls_dir = temp_dir / "htmls"
        
        (htmls_dir / "00_John Doe.html").write_text(sample_staff_page, encoding="utf-8")
        (htmls_dir / "01_Jane Smith.html").write_text("<html></html>", encoding="utf-8")
        
        df = process_all_data(staff_list, temp_dir)
        
        assert len(df) == 2
        assert df.loc[0, "year"] == "1985"
        assert pd.isna(df.loc[1, "year"])
        
        # Check if CSV was created
        csv_files = list(temp_dir.glob("ysu_staff_data_*.csv"))
        assert len(csv_files) == 1
    
    @patch('ysu_scraper.get_staff_list')
    @patch('ysu_scraper.download_all_pages')
    @patch('ysu_scraper.process_all_data')
    def test_run_full_scrape(self, mock_process, mock_download, mock_get_staff, temp_dir):
        """Test the complete scraping workflow."""
        # Setup mocks
        mock_staff_list = [{"name": "Test", "position": "Prof", "url": "/test", "img": "/test.jpg"}]
        mock_get_staff.return_value = mock_staff_list
        mock_download.return_value = [Path("test.html")]
        mock_df = pd.DataFrame(mock_staff_list)
        mock_process.return_value = mock_df
        
        result = run_full_scrape("http://example.com", save_dir=temp_dir)
        
        assert isinstance(result, pd.DataFrame)
        mock_get_staff.assert_called_once_with("http://example.com")
        mock_download.assert_called_once_with(mock_staff_list, n_jobs=4, save_dir=temp_dir)
        mock_process.assert_called_once_with(mock_staff_list, save_dir=temp_dir)
    
    @patch('ysu_scraper.Parallel')
    @patch('ysu_scraper.delayed')
    def test_download_all_pages(self, mock_delayed, mock_parallel, temp_dir):
        """Test parallel downloading of all pages."""
        # Mock the parallel execution
        mock_parallel_instance = Mock()
        mock_parallel.return_value = mock_parallel_instance
        mock_parallel_instance.return_value = [Path("file1.html"), None, Path("file3.html")]
        
        staff_list = [
            {"name": "Person 1", "url": "/1"},
            {"name": "Person 2", "url": "/2"},
            {"name": "Person 3", "url": "/3"}
        ]
        
        results = download_all_pages(staff_list, n_jobs=2, save_dir=temp_dir)
        
        assert len(results) == 3
        assert results[0] == Path("file1.html")
        assert results[1] is None
        assert results[2] == Path("file3.html")
        
        # Verify that Parallel was called with correct parameters
        mock_parallel.assert_called_once_with(n_jobs=2)


# Integration tests
class TestYSUScraperIntegration:
    """Integration tests for the scraper."""
    
    def test_real_html_parsing(self):
        """Test with more realistic HTML structure."""
        html = """
        <div class="view-content row">
            <div class="col-6 col-md-3 views-row">
                <a href="/faculty/staff/123">
                    <img src="/sites/default/files/styles/user_small/public/pictures/picture-123.jpg" />
                </a>
                <div class="user-full-name">Արմեն Սարգսյան</div>
                <div class="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item">
                    Պրոֆեսոր
                </div>
            </div>
        </div>
        """
        
        soup = BeautifulSoup(html, "html.parser")
        grid = soup.find("div", class_="view-content row")
        cells = grid.find_all("div", class_="col-6 col-md-3 views-row")
        
        assert len(cells) == 1
        
        cell = cells[0]
        url = cell.find("a")["href"]
        name = clean_name(cell.find("div", class_="user-full-name").text)
        position_elem = cell.find("div", class_="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item")
        position = position_elem.text.strip()
        
        assert url == "/faculty/staff/123"
        assert name == "Արմեն Սարգսյան"
        assert position == "Պրոֆեսոր"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
