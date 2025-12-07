# Fake News Detector - AI Classification

A Streamlit web application for detecting fake news using Large Language Models (LLMs) with few-shot learning, based on the OpenAI GPT API.

## Features

- **LLM-Powered Classification**: Uses OpenAI GPT models (GPT-3.5-turbo, GPT-4) with few-shot prompting
- **Few-Shot Learning**: Automatically creates balanced few-shot examples from training data
- **Configurable Prompting**: Adjustable number of examples per class in the prompt
- **Multiple Model Support**: Choose between different OpenAI models
- **Interactive Web Interface**: User-friendly Streamlit app for real-time classification
- **Batch Processing**: Support for multiple upload methods with cost estimation:
  - Text files (one article per line)
  - CSV files (with 'text' column)
  - Manual text input
- **Performance Evaluation**: Test model accuracy on held-out test set
- **Downloadable Results**: Export batch classification results as CSV
- **Visual Analytics**: Confusion matrix, metrics, and distribution charts

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector
cd fake-news-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset

You need a CSV file with news articles. Download from Kaggle or create your own:

**Option A: Download from Kaggle**
1. Go to [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download the dataset
3. Combine or prepare a CSV with these columns:
   - `text`: The news article content
   - `label`: Either "Fake" or "Real"

**Option B: Create Your Own**
Create a CSV file with two columns: `text` and `label`

Example:
```csv
text,label
"Breaking: Aliens land in Central Park",Fake
"President signs new climate legislation",Real
"Scientists discover cure for all diseases overnight",Fake
```

### 4. Add Your Dataset

Create a `data` folder and add your CSV:

```bash
mkdir data
# Copy your CSV file as news_data.csv
cp your_dataset.csv data/news_data.csv
```

### 5. Run the App Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Using the App

1. **Enter API Key**: Add your OpenAI API key in the sidebar
2. **Prepare Model**: Click "Prepare Few-Shot Learning Model" to set up
3. **Classify News**: 
   - Enter single articles in the text box
   - Upload files for batch processing
   - View results and metrics
4. **Evaluate**: Test model accuracy on your dataset
5. **Export**: Download results as CSV

## 🌐 Deploy to Streamlit Cloud

### Step 1: Fork the Repository

1. Go to this GitHub repository
2. Click "Fork" in the top right
3. This creates a copy in your GitHub account

### Step 2: Set Up Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Your forked repository
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"

### Step 3: Add Your Dataset to GitHub

```bash
# Make sure your data folder and CSV are tracked
git add data/news_data.csv
git commit -m "Add news dataset"
git push origin main
```

### Step 4: Configure Secrets (Optional)

If you want to pre-configure your API key:
1. In Streamlit Cloud, go to your app settings
2. Click "Secrets"
3. Add: `OPENAI_API_KEY = "your-api-key-here"`
4. Update the app code to read from secrets

### Step 5: Access Your App

Your app will be live at: `https://YOUR_USERNAME-fake-news-detector.streamlit.app`

## Project Structure

```
fake-news-detector/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── data/
    └── news_data.csv     # Your dataset (create this)
```

## Cost Estimates

- **GPT-3.5-turbo**: ~$0.002 per article
- **GPT-4**: ~$0.03 per article
- **Evaluation** (50 samples): $0.10 - $0.20

## 🛠️ Customization

### Change the Dataset Format

If your CSV has different column names, modify the `load_data()` function:

```python
df = df.rename(columns={'your_text_column': 'sentence', 'your_label_column': 'gold_label'})
```

### Adjust Few-Shot Examples

Change the slider in the sidebar or modify the default in the code:

```python
n_examples_per_class = st.sidebar.slider("Examples per class in prompt", min_value=1, max_value=30, value=5)
```

### Change Models

Add more models in the sidebar selectbox:

```python
model_choice = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"])
```

## Troubleshooting

**"Data file not found" error:**
- Make sure `data/news_data.csv` exists
- Check the file has correct columns: `text` and `label`

**API errors:**
- Verify your OpenAI API key is correct
- Check you have credits in your OpenAI account
- Ensure you're not hitting rate limits

**CSV encoding issues:**
- Try saving your CSV with UTF-8 encoding
- Use `encoding='utf-8'` when saving in pandas

## Dataset Sources

- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [Kaggle Fake News Detection](https://www.kaggle.com/c/fake-news)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

## Contributing

Feel free to fork this repository and submit pull requests!

## License

MIT License - feel free to use this for your projects!

## Academic Use

This project was created as part of a classification assignment. Feel free to adapt it for your own assignments, but make sure to:
- Use your own dataset
- Customize the classification task
- Add your own analysis

## Support

If you have questions:
1. Check this README
2. Review the Streamlit documentation
3. Check OpenAI API documentation
4. Open an issue on GitHub

---

Built with ❤️ using Streamlit and OpenAI GPT
