"""
Tests for YSU Faculty Staff Scraper

Run with: pytest test_ysu_scraper.py -v
"""

import pytest
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open
import pandas as pd

from ysu_scraper import YSUScraper


class TestYSUScraper:
    """Test cases for YSU Scraper functionality."""
    
    @pytest.fixture
    def temp_scraper(self):
        """Create a temporary scraper instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = YSUScraper(save_dir=temp_dir, log_level="ERROR")
            yield scraper
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing."""
        return """
        <div class="view-content row">
            <div class="col-6 col-md-3 views-row">
                <a href="/faculty/staff/123">
                    <img src="/images/john_doe.jpg" />
                </a>
                <div class="user-full-name">John Doe</div>
                <div class="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item">
                    Professor
                </div>
            </div>
            <div class="col-6 col-md-3 views-row">
                <a href="/faculty/staff/124">
                    <img src="/images/jane_smith.jpg" />
                </a>
                <div class="user-full-name">Jane Smith</div>
                <div class="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item">
                    Associate Professor
                </div>
            </div>
        </div>
        """
    
    @pytest.fixture
    def sample_personal_page(self):
        """Sample personal page HTML for testing."""
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
        assert YSUScraper.clean_name("  John  Doe  ") == "John Doe"
        assert YSUScraper.clean_name("John\nDoe") == "John Doe"
        assert YSUScraper.clean_name("John\n\nDoe  ") == "John Doe"
    
    def test_scraper_initialization(self, temp_scraper):
        """Test scraper initialization."""
        assert temp_scraper.base_url == "https://www.ysu.am"
        assert temp_scraper.save_dir.exists()
        assert (temp_scraper.save_dir / "htmls").exists()
    
    @patch('ysu_scraper.requests.get')
    def test_get_staff_list_success(self, mock_get, temp_scraper, sample_html):
        """Test successful staff list extraction."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = sample_html.encode('utf-8')
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the method
        staff_list = temp_scraper.get_staff_list("https://example.com/staff")
        
        assert len(staff_list) == 2
        assert staff_list[0]['name'] == 'John Doe'
        assert staff_list[0]['position'] == 'Professor'
        assert staff_list[0]['url'] == '/faculty/staff/123'
        assert staff_list[1]['name'] == 'Jane Smith'
        assert staff_list[1]['position'] == 'Associate Professor'
    
    @patch('ysu_scraper.requests.get')
    def test_get_staff_list_request_failure(self, mock_get, temp_scraper):
        """Test staff list extraction with request failure."""
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(requests.RequestException):
            temp_scraper.get_staff_list("https://example.com/staff")
    
    @patch('ysu_scraper.requests.get')
    def test_get_staff_list_no_grid(self, mock_get, temp_scraper):
        """Test staff list extraction when grid is not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body>No grid here</body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Staff grid not found"):
            temp_scraper.get_staff_list("https://example.com/staff")
    
    @patch('ysu_scraper.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_staff_page_success(self, mock_file, mock_get, temp_scraper):
        """Test successful staff page download."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html>Staff page content</html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        staff_info = {"name": "John Doe", "url": "/staff/123"}
        result = temp_scraper.download_staff_page(0, staff_info)
        
        assert result is not None
        assert "00_John Doe.html" in str(result)
        mock_file.assert_called_once()
    
    @patch('ysu_scraper.requests.get')
    def test_download_staff_page_failure(self, mock_get, temp_scraper):
        """Test staff page download failure."""
        mock_get.side_effect = requests.RequestException("Download error")
        
        staff_info = {"name": "John Doe", "url": "/staff/123"}
        result = temp_scraper.download_staff_page(0, staff_info)
        
        assert result is None
    
    def test_extract_year_of_birth_success(self, temp_scraper, sample_personal_page):
        """Test successful year of birth extraction."""
        soup = BeautifulSoup(sample_personal_page, "html.parser")
        year = temp_scraper.extract_year_of_birth(soup)
        
        assert year == "1985"
    
    def test_extract_year_of_birth_no_personal_info(self, temp_scraper):
        """Test year of birth extraction when personal info is missing."""
        soup = BeautifulSoup("<html><body>No personal info</body></html>", "html.parser")
        year = temp_scraper.extract_year_of_birth(soup)
        
        assert year is None
    
    def test_extract_year_of_birth_wrong_title(self, temp_scraper):
        """Test year of birth extraction with wrong title."""
        html = """
        <div class="personal-information">
            <div class="fw-bold d-flex field field--name-field-user-date-of-birth field--type-datetime field--label-above">
                Wrong Title
                1985
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        year = temp_scraper.extract_year_of_birth(soup)
        
        assert year is None
    
    @patch.object(YSUScraper, 'get_staff_list')
    @patch.object(YSUScraper, 'download_all_pages')
    @patch.object(YSUScraper, 'process_all_data')
    def test_run_full_scrape(self, mock_process, mock_download, mock_get_staff, temp_scraper):
        """Test the full scrape process."""
        # Mock the methods
        mock_staff_list = [{"name": "John Doe", "position": "Professor", "url": "/staff/123", "img": "/img.jpg"}]
        mock_get_staff.return_value = mock_staff_list
        mock_download.return_value = [Path("file1.html")]
        mock_df = pd.DataFrame(mock_staff_list)
        mock_process.return_value = mock_df
        
        # Run full scrape
        result_df = temp_scraper.run_full_scrape("https://example.com/staff")
        
        # Verify all methods were called
        mock_get_staff.assert_called_once_with("https://example.com/staff")
        mock_download.assert_called_once_with(mock_staff_list, n_jobs=4)
        mock_process.assert_called_once_with(mock_staff_list)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1


class TestIntegration:
    """Integration tests for the scraper."""
    
    @pytest.mark.slow
    @patch('ysu_scraper.requests.get')
    def test_real_website_structure(self, mock_get):
        """Test that the scraper handles the real YSU website structure."""
        # This would require actual HTML from YSU website
        # For now, we'll just test that it doesn't crash with realistic data
        
        realistic_html = """
        <html>
        <body>
            <div class="view-content row">
                <div class="col-6 col-md-3 views-row">
                    <a href="/hy/faculty/516/staff/18598">
                        <img src="/sites/default/files/styles/staff_thumb_167_200_/public/staff/18598.jpg" />
                    </a>
                    <div class="user-full-name">Արամ Մկրտչյան</div>
                    <div class="field field--name-field-user-intranet-position field--type-entity-reference field--label-hidden field__item">
                        ֆակուլտետի դեկան, տնօրեն, պրոֆեսոր
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = realistic_html.encode('utf-8')
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = YSUScraper(save_dir=temp_dir, log_level="ERROR")
            staff_list = scraper.get_staff_list("https://example.com/staff")
            
            assert len(staff_list) == 1
            assert staff_list[0]['name'] == 'Արամ Մկրտչյան'
            assert 'դեկան' in staff_list[0]['position']


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_staff_list(self):
        """Test processing with empty staff list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = YSUScraper(save_dir=temp_dir, log_level="ERROR")
            df = scraper.process_all_data([])
            
            assert len(df) == 0
            assert isinstance(df, pd.DataFrame)
    
    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = YSUScraper(save_dir=temp_dir, log_level="ERROR")
            soup = BeautifulSoup("<div><p>Malformed HTML</div>", "html.parser")
            year = scraper.extract_year_of_birth(soup)
            
            assert year is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
