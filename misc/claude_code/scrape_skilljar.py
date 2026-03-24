"""
Scrape Skilljar course: Introduction to Agent Skills (fully automated)
"""

import os
import re
import sys
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://anthropic.skilljar.com"


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text


def auto_login(page, email, password, course_url):
    """Automatically sign in to Skilljar."""
    page.goto(course_url, timeout=60000, wait_until="domcontentloaded")
    time.sleep(5)

    # Check if we need to log in
    current = page.url
    print(f"Current URL after navigation: {current}")

    # Look for sign-in / login links or forms
    # Skilljar may redirect to login or show a login button
    login_selectors = [
        "a:has-text('Sign In')",
        "a:has-text('Log In')",
        "a:has-text('Sign in')",
        "a:has-text('Log in')",
        "button:has-text('Sign In')",
        "button:has-text('Log In')",
        "a[href*='login']",
        "a[href*='signin']",
        "a[href*='auth']",
    ]

    for sel in login_selectors:
        btn = page.query_selector(sel)
        if btn and btn.is_visible():
            print(f"Clicking login button: {sel}")
            btn.click()
            page.wait_for_load_state("domcontentloaded")
            time.sleep(5)
            break

    print(f"URL after login click: {page.url}")

    # Fill email
    email_selectors = [
        "input[type='email']",
        "input[name='email']",
        "input[id='email']",
        "input[placeholder*='email' i]",
        "input[placeholder*='Email']",
    ]
    for sel in email_selectors:
        el = page.query_selector(sel)
        if el and el.is_visible():
            print(f"Filling email with selector: {sel}")
            el.fill(email)
            break

    # Fill password
    pw_selectors = [
        "input[type='password']",
        "input[name='password']",
        "input[id='password']",
    ]
    for sel in pw_selectors:
        el = page.query_selector(sel)
        if el and el.is_visible():
            print(f"Filling password with selector: {sel}")
            el.fill(password)
            break

    # Click submit
    submit_selectors = [
        "button[type='submit']",
        "input[type='submit']",
        "button:has-text('Sign In')",
        "button:has-text('Log In')",
        "button:has-text('Sign in')",
        "button:has-text('Log in')",
        "button:has-text('Submit')",
    ]
    for sel in submit_selectors:
        btn = page.query_selector(sel)
        if btn and btn.is_visible():
            print(f"Clicking submit: {sel}")
            btn.click()
            page.wait_for_load_state("domcontentloaded")
            time.sleep(5)
            break

    print(f"URL after login: {page.url}")


def extract_lesson_links(page, course_slug):
    """Find all lesson links in the course sidebar/curriculum."""
    time.sleep(3)

    selectors = [
        "a.sj-lesson-row",
        "#curriculum-list a",
        "#lp-left-nav a",
        ".sj-curriculum-item a",
        "a[href*='/page/']",
        ".lesson-row a",
        "a.curriculum-item",
    ]

    links = []
    seen_urls = set()

    for sel in selectors:
        elements = page.query_selector_all(sel)
        for el in elements:
            href = el.get_attribute("href")
            title = el.inner_text().strip()
            if href and title:
                # Match /course-slug/NUMERIC_ID or /page/ patterns
                is_lesson = (
                    "/page/" in href
                    or re.search(rf"/{course_slug}/\d+", href)
                )
                if is_lesson:
                    full_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        links.append((full_url, title))

    # Fallback: grab all anchors matching lesson patterns
    if not links:
        all_anchors = page.query_selector_all("a[href]")
        for el in all_anchors:
            href = el.get_attribute("href") or ""
            title = el.inner_text().strip()
            is_lesson = (
                "/page/" in href
                or re.search(rf"/{course_slug}/\d+", href)
            )
            if is_lesson and title:
                full_url = href if href.startswith("http") else f"{BASE_URL}{href}"
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    links.append((full_url, title))

    return links


def navigate_with_next_button(page):
    """Navigate through lessons using Next button."""
    lessons = []
    visited = set()

    while True:
        url = page.url
        if url in visited:
            break
        visited.add(url)

        title_el = page.query_selector("h1, .lesson-title, .sj-lesson-title")
        title = title_el.inner_text().strip() if title_el else f"lesson_{len(visited)}"
        lessons.append((url, title))
        print(f"  Found: {title} -> {url}")

        # Try clicking Next
        next_selectors = [
            "a.sj-next-lesson",
            "button.next-lesson",
            ".sj-next-btn",
            "a[rel='next']",
            "a:has-text('Next Lesson')",
            "a:has-text('Next')",
            "button:has-text('Next')",
            "a:has-text('Continue')",
        ]
        clicked = False
        for sel in next_selectors:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                page.wait_for_load_state("domcontentloaded")
                time.sleep(3)
                clicked = True
                break
        if not clicked:
            break

    return lessons


def scrape_lesson(page, url, title):
    """Navigate to a lesson page and extract its text content."""
    page.goto(url, timeout=60000, wait_until="domcontentloaded")
    time.sleep(3)

    content_selectors = [
        ".sj-lesson-content",
        ".lesson-content",
        "#lesson-content",
        "article",
        "main",
        ".content-wrapper",
        ".sj-content",
        "#content",
    ]

    text = ""
    for sel in content_selectors:
        el = page.query_selector(sel)
        if el:
            text = el.inner_text().strip()
            if len(text) > 50:
                break

    if len(text) < 50:
        text = page.query_selector("body").inner_text().strip()

    # Get code blocks
    code_blocks = page.query_selector_all("pre, code")
    code_texts = []
    for cb in code_blocks:
        ct = cb.inner_text().strip()
        if ct and len(ct) > 10:
            code_texts.append(ct)

    # Also try to get HTML content for better formatting
    html = ""
    for sel in content_selectors:
        el = page.query_selector(sel)
        if el:
            html = el.inner_html()
            if len(html) > 100:
                break

    return text, code_texts, html


def main():
    if len(sys.argv) < 5:
        print("Usage: python scrape_skilljar.py <email> <password> <course-slug> <output-dir>")
        print("Example: python scrape_skilljar.py user@ex.com pass123 claude-code-in-action claude_in_action")
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    course_slug = sys.argv[3]
    output_dir = os.path.join(os.path.dirname(__file__), sys.argv[4])
    course_url = f"{BASE_URL}/{course_slug}"

    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # Login
        print(f"Logging in and navigating to {course_url}...")
        auto_login(page, email, password, course_url)

        # Navigate to course page
        print("Navigating to course page...")
        page.goto(course_url, timeout=60000, wait_until="domcontentloaded")
        time.sleep(3)
        print(f"On page: {page.url}")

        # Debug: print all links on page
        print("\nAll links on page:")
        all_a = page.query_selector_all("a[href]")
        for a in all_a:
            href = a.get_attribute("href") or ""
            txt = a.inner_text().strip()[:80]
            if txt and (course_slug in href.lower() or "page" in href.lower() or "lesson" in href.lower()):
                print(f"  [{txt}] -> {href}")

        # Try to find lesson links
        print("\nExtracting lesson links...")
        lessons = extract_lesson_links(page, course_slug)

        if not lessons:
            print("No links found from course page. Trying to click into first lesson...")
            # Try clicking "Start" or "Begin" or first lesson
            start_selectors = [
                "a:has-text('Start')",
                "a:has-text('Begin')",
                "a:has-text('Resume')",
                "button:has-text('Start')",
                "button:has-text('Begin')",
                "button:has-text('Resume')",
                ".sj-start-btn",
                "a.sj-btn",
            ]
            for sel in start_selectors:
                btn = page.query_selector(sel)
                if btn and btn.is_visible():
                    print(f"Clicking: {sel}")
                    btn.click()
                    page.wait_for_load_state("domcontentloaded")
                    time.sleep(3)
                    break

            print(f"Now on: {page.url}")

            # Try extracting links from lesson page sidebar
            lessons = extract_lesson_links(page, course_slug)

            if not lessons:
                print("Navigating via Next button...")
                lessons = navigate_with_next_button(page)

        print(f"\nFound {len(lessons)} lessons total:")
        for i, (url, title) in enumerate(lessons, 1):
            print(f"  {i}. {title}")

        # Scrape each lesson
        for i, (url, title) in enumerate(lessons, 1):
            print(f"\nScraping [{i}/{len(lessons)}]: {title}...")
            text, code_blocks, html = scrape_lesson(page, url, title)

            filename = f"{i:02d}_{slugify(title)[:60]}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n")
                f.write(f"# Source: {url}\n")
                f.write("=" * 60 + "\n\n")
                f.write(text)
                if code_blocks:
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write("# CODE BLOCKS\n")
                    f.write("=" * 60 + "\n")
                    for j, cb in enumerate(code_blocks, 1):
                        f.write(f"\n--- Code Block {j} ---\n")
                        f.write(cb + "\n")

            # Also save HTML version
            html_filepath = filepath.replace(".txt", ".html")
            with open(html_filepath, "w", encoding="utf-8") as f:
                f.write(f"<h1>{title}</h1>\n<p>Source: {url}</p>\n<hr>\n")
                f.write(html)

            print(f"  Saved: {filename} ({len(text)} chars)")

        browser.close()

    print("\n" + "=" * 60)
    print(f"DONE! Scraped {len(lessons)} lessons to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
