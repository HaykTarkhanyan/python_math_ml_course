# HTML & CSS Examples for Web Scraping

This folder contains HTML and CSS examples used in the web scraping tutorial.

## Files Included:

### 1. `sample_page.html`
A complete sample website with typical blog-style content including:
- Header with navigation
- Article posts with metadata
- CSS styling for visual appeal
- Data attributes for scraping practice

**Key elements for scraping practice:**
- Articles with class="post"
- Author and date information
- Navigation links
- Data attributes (data-id)

### 2. `product_page.html`
An e-commerce style page featuring:
- Product listings with prices
- Rating and review information
- Product categories (data-category attribute)
- Blog post section

**Scraping targets:**
- Product names and descriptions
- Prices and ratings
- Review counts
- Category information

### 3. `crypto_table.html`
A cryptocurrency market data table with:
- Structured table data
- Positive/negative price changes
- Market capitalization data
- Styled with dark theme

**Table scraping practice:**
- Extract headers and data rows
- Handle financial data formatting
- Process percentage changes
- Work with large numbers (billions)

### 4. `css_examples.css`
Comprehensive CSS reference showing:
- All major selector types
- Pseudo-classes and pseudo-elements
- Responsive design patterns
- CSS Grid and Flexbox
- CSS Variables

## Usage in Web Scraping:

### CSS Selectors for Beautiful Soup:
```python
# Element selector
soup.select('p')

# Class selector
soup.select('.post-title')

# ID selector
soup.select('#main-header')

# Attribute selector
soup.select('[data-category="electronics"]')

# Descendant selector
soup.select('.navigation ul li')

# Child selector
soup.select('.content > article')
```

### Common Scraping Patterns:

1. **Extract all articles:**
   ```python
   articles = soup.find_all('article', class_='post')
   ```

2. **Get product information:**
   ```python
   products = soup.find_all('div', class_='product')
   for product in products:
       name = product.h3.text
       price = product.get('data-price')
   ```

3. **Table data extraction:**
   ```python
   table = soup.find('table', id='crypto-table')
   rows = table.find('tbody').find_all('tr')
   ```

## Best Practices:

1. **Always inspect the HTML structure** before writing scrapers
2. **Use specific selectors** to avoid false matches
3. **Handle missing elements** with proper error checking
4. **Respect the website's structure** and don't rely on styling classes that might change

## Testing Your Scrapers:

You can use these HTML files to test your scraping code locally:

```python
from bs4 import BeautifulSoup

# Read local HTML file
with open('sample_page.html', 'r', encoding='utf-8') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')
# Now practice your scraping techniques!
```

This approach allows you to develop and test your scraping logic without making repeated requests to live websites.
