// Homework JavaScript Functionality
// This file contains all the JavaScript needed for homework files

// Function to add cheese emojis based on difficulty attribute
function addCheeseEmojis() {
  // Find all h2 elements (problem headers)
  const headers = document.querySelectorAll('h2');
  const maxCheese = 3; // Maximum number of cheese emojis
  
  headers.forEach(header => {
    // Check if header has data-difficulty attribute
    const cheeseCount = header.getAttribute('data-difficulty');
    if (cheeseCount && !isNaN(cheeseCount) && parseInt(cheeseCount) > 0) {
      const count = Math.min(parseInt(cheeseCount), maxCheese); // Cap at maxCheese
      const cheeseEmojis = '🧀'.repeat(count);
      
      // Add cheese emojis as prefix if not already present
      if (!header.textContent.startsWith('🧀')) {
        header.innerHTML = cheeseEmojis + ' ' + header.innerHTML;
      }
    }
  });
}

// Initialize all functionality when page loads
document.addEventListener('DOMContentLoaded', function() {
  // Add cheese emojis to problem headers
  addCheeseEmojis();
});
