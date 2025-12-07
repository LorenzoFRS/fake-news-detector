# Deployment Guide

Here's how to get your app online. Should take about 15-20 minutes.

## What you need

- GitHub account
- OpenAI API key
- Your dataset (CSV with 'text' and 'label' columns)

## Getting your code on GitHub

If it's not already there:

1. Make a new repo on GitHub (name it whatever, just make it public)
2. In your project folder, run:

```bash
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git
git branch -M main
git push -u origin main
```

You'll need a Personal Access Token instead of your password. Get one at github.com/settings/tokens and make sure to check the "repo" box.

## About the dataset

Your CSV needs two columns: `text` and `label` (either "Fake" or "Real").

Quick ways to get data:
- Run the sample script: `python create_sample_dataset.py`
- Download from Kaggle (just search "fake news dataset")
- Make your own

If your file is huge (over 100MB), you'll need to make it smaller since GitHub has limits:

```python
import pandas as pd
df = pd.read_csv('big_file.csv')
df = df.sample(5000)
df.to_csv('data/news_data.csv', index=False)
```

## Deploying on Streamlit

1. Go to share.streamlit.io and sign in with GitHub
2. Click "New app"
3. Pick your repo from the dropdown
4. Set branch to "main" and file to "app.py"
5. Click Deploy

It takes a few minutes to build. You'll see logs showing what's happening. If something breaks, it'll show up there.

Your app URL will be something like `https://your-username-fake-news-detector.streamlit.app`

## If you want to hide your API key

In your app settings on Streamlit Cloud:
1. Go to Settings > Secrets
2. Add: `OPENAI_API_KEY = "your-key-here"`
3. Update your code to check for it first

## Common problems

**"Module not found"**
Add the missing package to requirements.txt

**"File not found"**
Check your file paths. Make sure data/news_data.csv actually exists in your repo.

**App is slow or crashes**
Your dataset might be too big. Try using fewer rows.

**API errors**
Double check your key works and you have credits

## Updating after deployment

When you want to change something:
1. Make your changes
2. Test locally with `streamlit run app.py`
3. Push to GitHub:
```bash
git add .
git commit -m "updated stuff"
git push
```

Streamlit auto-redeploys when you push.

## Keeping costs down

Since you're paying for API calls:
- Use GPT-3.5-turbo instead of GPT-4
- Start with fewer examples (like 3-5)
- Test on small batches first
- Set spending limits on OpenAI

For reference, 100 articles with GPT-3.5 costs about 20 cents.

## Testing your deployed app

Once it's live, make sure to test:
- Can you enter your API key?
- Does preparing the model work?
- Can you classify a single article?
- Can you upload a file?
- Does download work?

## If things break

Check the logs in Streamlit Cloud first. Most issues are either:
- Missing packages in requirements.txt
- Wrong file paths
- API problems

