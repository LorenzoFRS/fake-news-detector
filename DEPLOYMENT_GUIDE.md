Deployment Guide
Here's how to get your app online. Should take about 15-20 minutes.
What you need

GitHub account
OpenAI API key
Your dataset (CSV with 'text' and 'label' columns)

Getting your code on GitHub
If it's not already there:

Make a new repo on GitHub (name it whatever, just make it public)
In your project folder, run:

bashgit init
git add .
git commit -m "first commit"
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git
git branch -M main
git push -u origin main
You'll need a Personal Access Token instead of your password. Get one at github.com/settings/tokens and make sure to check the "repo" box.
About the dataset
Your CSV needs two columns: text and label (either "Fake" or "Real").
Quick ways to get data:

Run the sample script: python create_sample_dataset.py
Download from Kaggle (just search "fake news dataset")
Make your own

If your file is huge (over 100MB), you'll need to make it smaller since GitHub has limits:
pythonimport pandas as pd
df = pd.read_csv('big_file.csv')
df = df.sample(5000)
df.to_csv('data/news_data.csv', index=False)
Deploying on Streamlit

Go to share.streamlit.io and sign in with GitHub
Click "New app"
Pick your repo from the dropdown
Set branch to "main" and file to "app.py"
Click Deploy

It takes a few minutes to build. You'll see logs showing what's happening. If something breaks, it'll show up there.
Your app URL will be something like https://your-username-fake-news-detector.streamlit.app
If you want to hide your API key
In your app settings on Streamlit Cloud:

Go to Settings > Secrets
Add: OPENAI_API_KEY = "your-key-here"
Update your code to check for it first

Common problems
"Module not found"
Add the missing package to requirements.txt
"File not found"
Check your file paths. Make sure data/news_data.csv actually exists in your repo.
App is slow or crashes
Your dataset might be too big. Try using fewer rows.
API errors
Double check your key works and you have credits
Updating after deployment
When you want to change something:

Make your changes
Test locally with streamlit run app.py
Push to GitHub:

bashgit add .
git commit -m "updated stuff"
git push
Streamlit auto-redeploys when you push.
Keeping costs down
Since you're paying for API calls:

Use GPT-3.5-turbo instead of GPT-4
Start with fewer examples (like 3-5)
Test on small batches first
Set spending limits on OpenAI

For reference, 100 articles with GPT-3.5 costs about 20 cents.
Testing your deployed app
Once it's live, make sure to test:

Can you enter your API key?
Does preparing the model work?
Can you classify a single article?
Can you upload a file?
Does download work?

What to submit
For the assignment you'll probably need:

App URL
GitHub repo URL
Maybe a quick description of what you did

If things break
Check the logs in Streamlit Cloud first. Most issues are either:

Missing packages in requirements.txt
Wrong file paths
API problems

If you're really stuck, try redeploying from scratch.
Privacy stuff
Your app is public, so anyone with the link can use it. Don't put API keys in your code. Use the secrets thing in Streamlit if you need to.
Quick checklist
Before you call it done:

App loads without errors
You can classify articles
Batch upload works
GitHub repo looks good
README makes sense
