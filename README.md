# Does Welsh Media Need a Review? Detecting Sentiment Bias in Political Reporting Ahead of the 2026 Senedd Election

## Research Question

Does Welsh online media exhibit systematic sentiment bias toward or against political parties in the run-up to the 2026 Senedd election?

## Background

Ahead of the May 2026 Senedd election — widely described as the most consequential since devolution — accusations of political bias in Welsh media have come from multiple directions. This project uses NLP techniques to empirically examine whether measurable sentiment bias exists in Welsh political reporting, and whether it differs across outlets.

## Data Sources

Articles scraped from:
- **BBC Wales** (bbc.co.uk/news/wales)
- **Nation.Cymru** (nation.cymru)
- **WalesOnline** (walesonline.co.uk)

Coverage period: September 2025 – March 2026

## Method

- Sentiment analysis of text surrounding political party mentions
- Coverage volume and prominence comparison across parties and outlets
- Named entity recognition for party/leader identification

## Reproducing Results

```bash
pip install -r requirements.txt
python scripts/scrape.py
python scripts/preprocess.py
python scripts/analyse.py
jupyter notebook notebook.ipynb
```
