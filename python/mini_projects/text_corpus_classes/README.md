# 📚 Text Corpus Analyzer - OOP Project

A data-driven Object-Oriented Programming project that analyzes literary works from great 20th-century authors.

## 📁 Project Structure

```
text_corpus_classes/
├── data/
│   └── authors.json       # All authors in one file
├── main.ipynb             # Main notebook with classes and analysis
└── README.md              # This file
```

## 🎯 Learning Objectives

This project teaches:
- **Classes and Objects**: Creating and using custom classes
- **Inheritance**: Understanding class hierarchies
- **Composition**: Using objects within objects
- **Data Analysis**: Working with real literary data
- **Text Processing**: Tokenization, frequency analysis
- **Algorithms**: TF-IDF, similarity metrics, sentiment analysis

## 🏗️ Classes

### 📝 Document Class
Represents a single text excerpt with analysis methods:
- Word counting and tokenization
- Vocabulary richness
- Average word and sentence length
- Word frequency analysis
- Word search

### ✍️ Author Class
Represents an author with their collection of documents:
- Aggregate statistics across all documents
- Favorite words identification
- Theme distribution
- Document searching

### 📚 Corpus Class
Manages the entire collection of documents:
- Multi-author comparisons
- Theme-based searching
- Word search across all documents

## 📊 Analysis Features

1. **Word Statistics**: Count, frequency, vocabulary richness
2. **Author Comparison**: Compare writing styles (word length, sentence length, vocabulary)
3. **Theme Analysis**: Group and analyze documents by theme
4. **Search**: Find documents containing specific words
5. **Frequency Analysis**: Identify most common words per author

## 🚀 Getting Started

```python
# Create a corpus
corpus = Corpus("Great 20th Century Writers")

# Load data (all authors in one file)
corpus.load_from_json('data/authors.json')

# Analyze!
print(corpus.summary())
```

## 📚 Featured Authors

- **Fyodor Dostoevsky** (1821-1881) - Russian psychological realist
  - Works: Crime and Punishment, The Brothers Karamazov, Notes from Underground
  
- **Albert Camus** (1913-1960) - French absurdist philosopher
  - Works: The Stranger, The Myth of Sisyphus, The Plague
  
- **Erich Maria Remarque** (1898-1970) - German war novelist
  - Works: All Quiet on the Western Front, Three Comrades, Arch of Triumph

## 🎓 Extension Ideas

1. Add more authors (Tolstoy, Kafka, Hemingway)
2. Implement bigram/trigram analysis
3. Create visualization with matplotlib
4. Add more sophisticated sentiment analysis
5. Build a recommendation system
6. Analyethods to find longest words
5. Analyze stylistic evolution over time
6. Create specialized document subclasses (NovelExcerpt, PhilosophicalText)
7. Export results to CSV
8. Create a Theme class for theme-based analysis

## 🛠️ Requirements

- Python 3.7+
- Standard library only (no external packages required!)

## 📖 Educational Use

This project is designed for students learning OOP in Python. It demonstrates:
- **Classes and objects**: Creating instances with attributes and methods
- **Composition**: Author contains Documents, Corpus contains Authors
- **Data processing**: Working with real literary texts
- **Practical analysis**: Produces interesting, real insights about famous authors

Perfect for university courses teaching Object-Oriented Programming