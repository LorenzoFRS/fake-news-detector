# Fake News Detector

This is a web app that uses GPT to classify news articles as real or fake. I built it using Streamlit and OpenAI's API.

## What it does

The app takes news articles and predicts whether they're real news or fake news. It uses few-shot learning, which basically means showing GPT some examples first so it knows what to look for.

## Setup

You'll need Python 3.8+ and an OpenAI API key.

```bash
git clone https://github.com/LorenzoFRS/fake-news-detector
cd fake-news-detector
pip install -r requirements.txt
```

## Getting a dataset

The app needs a CSV file with two columns: `text` (the article) and `label` (either "Fake" or "Real").

You can either:
- Run `python create_sample_dataset.py` to make some sample data
- Download a dataset from Kaggle (search for "fake news dataset")
- Make your own CSV with news articles you find

Save it as `data/news_data.csv`

## Running it

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How to use it

1. Put in your OpenAI API key (get one from platform.openai.com)
2. Click the button to prepare the model
3. Type in an article or upload a file to classify multiple at once

## Features

- Classify single articles
- Upload files for batch processing (txt or csv)
- See accuracy metrics on test data
- Download results as CSV

## Deployment

To put this online, push your code to GitHub then:
1. Go to share.streamlit.io
2. Sign in with GitHub
3. Click New app and select your repo
4. Wait a few minutes for it to deploy

Your app will get a URL like `https://your-username-fake-news-detector.streamlit.app`

## Settings you can change

In the sidebar you can adjust:
- How many example articles to show the model (more = better but slower)
- Which GPT model to use (3.5-turbo is cheaper, 4 is more accurate)
- Test set size for evaluation

## About costs

This uses the OpenAI API which costs money. Rough estimates:
- GPT-3.5-turbo: around $0.002 per article
- GPT-4: around $0.03 per article

The app shows estimates before you run anything expensive.

## Project structure

```
fake-news-detector/
├── app.py                   # main app code
├── requirements.txt         # packages needed
├── create_sample_dataset.py # makes sample data
└── data/
    └── news_data.csv       # your dataset
```

## Notes

- Classification isn't perfect, especially for satire or complex cases
- Quality depends a lot on your dataset
- There are rate limits on the API depending on your account

## Issues I ran into

If you get "data file not found", make sure the CSV is in the data folder with the right columns.

If you get API errors, double check your key and that you have credits.

## Assignment context

This was built for a text classification assignment where we had to adapt an existing app to solve a different problem. I picked fake news detection because it seemed relevant and interesting.

## License

MIT - do whatever you want with it.
