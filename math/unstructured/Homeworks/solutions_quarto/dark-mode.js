// Dark Mode Toggle Functionality
// Author: GitHub Copilot
// Purpose: Centralized dark mode functionality for all homework solution documents

// Dark mode toggle function
function toggleDarkMode() {
  const html = document.documentElement;
  const themeIcon = document.getElementById('theme-icon');
  
  if (html.getAttribute('data-theme') === 'dark') {
    html.removeAttribute('data-theme');
    themeIcon.textContent = '🌙';
    localStorage.setItem('theme', 'light');
  } else {
    html.setAttribute('data-theme', 'dark');
    themeIcon.textContent = '☀️';
    localStorage.setItem('theme', 'dark');
  }
}

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function() {
  const savedTheme = localStorage.getItem('theme');
  const themeIcon = document.getElementById('theme-icon');
  
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeIcon.textContent = '☀️';
  } else {
    themeIcon.textContent = '🌙';
  }
});
