import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import openai
import random
import time
from typing import List, Dict

def load_data():
    """Load news data from CSV file."""
    data_file = "data/news_data.csv"
    
    try:
        # Read the CSV file
        df = pd.read_csv(data_file, encoding='utf-8')
        
        # Check if required columns exist
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("CSV file must contain 'text' and 'label' columns")
            return pd.DataFrame()
        
        # Rename columns to match our app structure
        df = df.rename(columns={'text': 'sentence', 'label': 'gold_label'})
        
        # Filter out any invalid labels (keep only 'Fake' and 'Real')
        df = df[df['gold_label'].isin(['Fake', 'Real'])]
        
        # Sample data if too large (to keep app responsive)
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
        
        return df
        
    except FileNotFoundError:
        st.error(f"Data file not found: {data_file}")
        st.info("Please upload a CSV file with 'text' and 'label' columns to data/news_data.csv")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_balanced_split(df, test_size=0.3, random_state=42):
    """Create a balanced train-test split for few-shot examples and evaluation."""
    X = df['sentence']
    y = df['gold_label']
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_few_shot_examples(X_train, y_train, n_examples_per_class=10):
    """Create balanced few-shot examples for the LLM prompt."""
    df_train = pd.DataFrame({'sentence': X_train, 'label': y_train})
    
    few_shot_examples = []
    
    # Get examples for each class
    for label in ['Real', 'Fake']:
        class_examples = df_train[df_train['label'] == label].sample(
            n=min(n_examples_per_class, len(df_train[df_train['label'] == label])),
            random_state=42
        )
        
        for _, row in class_examples.iterrows():
            few_shot_examples.append({
                'sentence': row['sentence'],
                'label': row['label']
            })
    
    # Shuffle the examples
    random.shuffle(few_shot_examples)
    return few_shot_examples

def build_few_shot_prompt(few_shot_examples: List[Dict], target_text: str) -> str:
    """Build a few-shot prompt for the LLM."""
    
    prompt = """You are a fake news detection system that classifies news articles as either "Real" or "Fake".

Your task is to analyze the news text and determine if it's legitimate news (Real) or fake/misleading news (Fake).

Consider these factors:
- Sensational or clickbait language suggests fake news
- Verifiable facts and neutral tone suggest real news
- Emotional manipulation or conspiracy theories suggest fake news
- Credible sources and balanced reporting suggest real news

Here are some examples:

"""
    
    # Add few-shot examples
    for example in few_shot_examples:
        # Truncate long examples to keep prompt manageable
        text_snippet = example["sentence"][:300] + "..." if len(example["sentence"]) > 300 else example["sentence"]
        prompt += f'Text: "{text_snippet}"\nClassification: {example["label"]}\n\n'
    
    # Add the target text
    target_snippet = target_text[:500] + "..." if len(target_text) > 500 else target_text
    prompt += f'Text: "{target_snippet}"\nClassification:'
    
    return prompt

def predict_with_llm(client, few_shot_examples: List[Dict], texts: List[str], model_name: str = "gpt-4-turbo-preview") -> List[Dict]:
    """Make predictions using LLM with few-shot prompting."""
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    
    for text in texts:
        try:
            # Clean and normalize text to handle unicode characters
            cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
            if not cleaned_text.strip():
                # If text becomes empty after cleaning, use original with replacement
                cleaned_text = text.encode('ascii', 'replace').decode('ascii')
            
            # Build the prompt
            prompt = build_few_shot_prompt(few_shot_examples, cleaned_text)
            
            # Make API call
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful fake news detection system. Respond with exactly 'Real' or 'Fake' only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Clean up the prediction
            if "Fake" in prediction:
                prediction = "Fake"
            elif "Real" in prediction:
                prediction = "Real"
            else:
                # Default to Real if unclear
                prediction = "Real"
            
            result = {
                'text': text,
                'prediction': prediction
            }
            
            results.append(result)
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing text: {e}")
            # Return default result on error
            result = {
                'text': text,
                'prediction': "Real"
            }
            results.append(result)
    
    return results

def evaluate_llm_model(client, few_shot_examples: List[Dict], X_test, y_test, model_name: str = "gpt-3.5-turbo"):
    """Evaluate the LLM model on test set."""
    # Sample a smaller subset for evaluation to save costs
    test_sample_size = min(50, len(X_test))
    test_indices = random.sample(range(len(X_test)), test_sample_size)
    
    X_test_sample = [X_test.iloc[i] for i in test_indices]
    y_test_sample = [y_test.iloc[i] for i in test_indices]
    
    # Get predictions
    predictions = predict_with_llm(client, few_shot_examples, X_test_sample, model_name)
    y_pred = [pred['prediction'] for pred in predictions]
    
    return y_test_sample, y_pred, X_test_sample

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("Fake News Detector - AI Classification")
    st.markdown("*Powered by OpenAI GPT with Few-Shot Learning*")
    st.markdown("Detect fake news and misinformation using advanced AI language models")
    st.markdown("---")
    
    # API Configuration
    st.sidebar.header("API Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    model_choice = st.sidebar.selectbox(
        "Model", 
        ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4"], 
        index=0,
        help="GPT-3.5-turbo is faster and cheaper, GPT-4 is more accurate"
    )
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.info("""
        **Don't have an API key?**
        1. Go to [platform.openai.com](https://platform.openai.com)
        2. Sign up or log in
        3. Navigate to API Keys section
        4. Create a new API key
        """)
        return
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()
    
    if df.empty:
        st.error("No data available. Please check the data files.")
        st.info("""
        **To use this app, you need a dataset:**
        1. Go to [Kaggle Fake News Datasets](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
        2. Download the dataset
        3. Create a CSV file with columns: 'text' and 'label' (values: 'Fake' or 'Real')
        4. Save it as `data/news_data.csv` in your project directory
        """)
        return
    
    # Sidebar with dataset info
    st.sidebar.header("Dataset Information")
    st.sidebar.metric("Total Articles", len(df))
    
    class_counts = df['gold_label'].value_counts()
    for label, count in class_counts.items():
        emoji = "✅" if label == "Real" else "❌"
        st.sidebar.metric(f"{emoji} {label} News", count)
    
    # Configuration
    st.sidebar.header("Model Configuration")
    n_examples_per_class = st.sidebar.slider(
        "Examples per class in prompt", 
        min_value=1, 
        max_value=30, 
        value=5, 
        help="Number of examples for each class (Real/Fake) to include in the few-shot prompt"
    )
    test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=30) / 100
    
    # Main content
    st.header("Model Setup")
    
    if st.button("🔄 Prepare Few-Shot Learning Model", type="primary"):
        with st.spinner("Preparing few-shot examples and splitting data..."):
            # Create train-test split
            X_train, X_test, y_train, y_test = create_balanced_split(df, test_size=test_size)
            
            # Create few-shot examples
            few_shot_examples = create_few_shot_examples(X_train, y_train, n_examples_per_class)
            
            # Store in session state
            st.session_state.client = client
            st.session_state.few_shot_examples = few_shot_examples
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.model_name = model_choice
            
            st.success(f"Model preparation complete!")
            
            # Show few-shot examples info
            st.info(f"""
            **Few-Shot Examples**: {len(few_shot_examples)} total
            - {len([ex for ex in few_shot_examples if ex['label'] == 'Real'])} Real news examples
            - {len([ex for ex in few_shot_examples if ex['label'] == 'Fake'])} Fake news examples
            
            **Test Set**: {len(X_test)} examples  
            - Real: {sum(y_test == 'Real')}
            - Fake: {sum(y_test == 'Fake')}
            """)
            
            # Show sample few-shot examples
            with st.expander("View Sample Few-Shot Examples"):
                for i, example in enumerate(few_shot_examples[:6]):  # Show first 6
                    if example['label'] == 'Fake':
                        st.error(f"**❌ {example['label']}**: {example['sentence'][:200]}...")
                    else:
                        st.success(f"**✅ {example['label']}**: {example['sentence'][:200]}...")
                
                # Show example prompt
                st.markdown("---")
                st.subheader("Example Full Prompt")
                example_prompt = build_few_shot_prompt(few_shot_examples, "This is an example news article for demonstration purposes.")
                st.code(example_prompt, language="text")
    
    st.markdown("---")
    
    # Show LLM evaluation if available
    if 'few_shot_examples' in st.session_state:
        st.header("📊 Model Performance Evaluation")
        
        with st.expander("Evaluate Model on Test Set", expanded=False):
            st.warning(f"This will make ~50 API calls. Estimated cost: ~$0.05-0.20 for {st.session_state.model_name}")
            
            if st.button("Run Model Evaluation"):
                with st.spinner("Evaluating model performance on test set..."):
                    try:
                        y_test_sample, y_pred, X_test_sample = evaluate_llm_model(
                            st.session_state.client,
                            st.session_state.few_shot_examples,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            st.session_state.model_name
                        )
                        
                        # Store evaluation results
                        st.session_state.y_test_sample = y_test_sample
                        st.session_state.y_pred = y_pred
                        st.session_state.X_test_sample = X_test_sample
                        
                        # Calculate accuracy
                        accuracy = accuracy_score(y_test_sample, y_pred)
                        st.success(f"Evaluation Complete! Test Accuracy: {accuracy:.1%}")
                        
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
        
        # Show evaluation results if available
        if 'y_test_sample' in st.session_state and 'y_pred' in st.session_state:
            st.subheader("Evaluation Results")
            
            # Classification report
            try:
                report = classification_report(
                    st.session_state.y_test_sample, 
                    st.session_state.y_pred, 
                    output_dict=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Classification Metrics")
                    # Create metrics dataframe
                    metrics_data = {
                        'Real News': {
                            'Precision': report.get('Real', {}).get('precision', 0),
                            'Recall': report.get('Real', {}).get('recall', 0), 
                            'F1-Score': report.get('Real', {}).get('f1-score', 0),
                        },
                        'Fake News': {
                            'Precision': report.get('Fake', {}).get('precision', 0),
                            'Recall': report.get('Fake', {}).get('recall', 0),
                            'F1-Score': report.get('Fake', {}).get('f1-score', 0),
                        }
                    }
                    metrics_df = pd.DataFrame(metrics_data).T
                    st.dataframe(metrics_df.round(3))
                    
                    st.metric("Overall Accuracy", f"{report['accuracy']:.1%}")
                
                with col2:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(
                        st.session_state.y_test_sample, 
                        st.session_state.y_pred,
                        labels=['Real', 'Fake']
                    )
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        labels={'x': 'Predicted', 'y': 'Actual'},
                        x=['Real', 'Fake'],
                        y=['Real', 'Fake'],
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig.update_layout(
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error generating evaluation metrics: {e}")
                
            # Show sample predictions
            st.subheader("Sample Predictions")
            sample_df = pd.DataFrame({
                'Actual': st.session_state.y_test_sample,
                'Predicted': st.session_state.y_pred,
                'Text': [text[:150] + "..." if len(text) > 150 else text 
                        for text in st.session_state.X_test_sample]
            })
            
            # Add correctness column
            sample_df['Correct'] = sample_df['Actual'] == sample_df['Predicted']
            
            # Color code correct/incorrect predictions
            def highlight_predictions(row):
                if row['Correct']:
                    return ['background-color: #1d3a1d'] * len(row)  # Dark green for correct
                else:
                    return ['background-color: #4d0a0a'] * len(row)  # Dark red for incorrect
            
            st.dataframe(
                sample_df.style.apply(highlight_predictions, axis=1),
                use_container_width=True,
                height=400
            )
    
    st.markdown("---")
    
    st.header("Detect Fake News")
    
    if 'few_shot_examples' not in st.session_state:
        st.warning("Please prepare the few-shot learning model first!")
    else:
        # Single text prediction
        st.subheader("Single Article Classification")
        user_text = st.text_area(
            "Enter a news article or headline to classify:", 
            placeholder="Paste a news article, headline, or statement here...",
            height=150
        )
        
        if st.button("Analyze Article", type="primary") and user_text:
            with st.spinner("Analyzing with AI..."):
                results = predict_with_llm(
                    st.session_state.client, 
                    st.session_state.few_shot_examples, 
                    [user_text],
                    st.session_state.model_name
                )
                result = results[0]
            
            # Display prediction with styling
            if result['prediction'] == 'Fake':
                st.error("### ❌ FAKE NEWS DETECTED")
                st.warning("This article shows characteristics of fake or misleading news.")
            else:
                st.success("### ✅ LIKELY REAL NEWS")
                st.info("This article appears to be legitimate news content.")
        
        st.markdown("---")
        
        # Batch upload
        st.subheader("Batch Analysis")
        st.info("**Tip**: Process multiple articles at once for efficiency")
        
        # File upload methods
        upload_method = st.radio("Choose input method:", ["Text File", "CSV File", "Manual Input"])
        
        if upload_method == "Text File":
            uploaded_file = st.file_uploader(
                "Upload a text file (one article per line)", 
                type=['txt'],
                help="Each line should contain one news article or headline"
            )
            if uploaded_file is not None:
                texts = uploaded_file.read().decode('utf-8').strip().split('\n')
                texts = [t.strip() for t in texts if t.strip()]
                
                st.info(f"Found {len(texts)} articles. Estimated cost: ~${len(texts) * 0.002:.3f}")
                
                if st.button("Analyze All Articles"):
                    process_batch_llm(texts)
        
        elif upload_method == "CSV File":
            uploaded_file = st.file_uploader(
                "Upload a CSV file with a 'text' or 'article' column", 
                type=['csv'],
                help="CSV must contain a column named 'text' or 'article'"
            )
            if uploaded_file is not None:
                try:
                    csv_df = pd.read_csv(uploaded_file)
                    
                    # Check for text column (flexible naming)
                    text_col = None
                    for col in ['text', 'article', 'content', 'news']:
                        if col in csv_df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        texts = csv_df[text_col].dropna().tolist()
                        st.info(f"Found {len(texts)} articles in column '{text_col}'. Estimated cost: ~${len(texts) * 0.002:.3f}")
                        
                        if st.button("Analyze All Articles"):
                            process_batch_llm(texts)
                    else:
                        st.error("CSV file must contain a 'text', 'article', 'content', or 'news' column")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
        
        else:  # Manual Input
            manual_texts = st.text_area(
                "Enter multiple articles (one per line):", 
                height=200,
                placeholder="Article 1...\nArticle 2...\nArticle 3..."
            )
            
            if st.button("🔍 Analyze Articles") and manual_texts:
                texts = [t.strip() for t in manual_texts.strip().split('\n') if t.strip()]
                st.info(f"Processing {len(texts)} articles. Estimated cost: ~${len(texts) * 0.002:.3f}")
                process_batch_llm(texts)

def process_batch_llm(texts):
    """Process a batch of texts using LLM and display results."""
    if not texts:
        st.warning("No texts to classify!")
        return
    
    with st.spinner(f"Analyzing {len(texts)} articles with AI... This may take a few minutes."):
        results = predict_with_llm(
            st.session_state.client,
            st.session_state.few_shot_examples,
            texts,
            st.session_state.model_name
        )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    st.subheader("Batch Analysis Results")
    
    fake_count = sum(1 for r in results if r['prediction'] == 'Fake')
    real_count = len(results) - fake_count
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyzed", len(results))
    with col2:
        st.metric("❌ Fake News", fake_count, delta=f"{fake_count/len(results)*100:.1f}%")
    with col3:
        st.metric("✅ Real News", real_count, delta=f"{real_count/len(results)*100:.1f}%")
    
    # Visualization
    st.subheader("Distribution")
    fig = px.pie(
        names=['Fake News', 'Real News'],
        values=[fake_count, real_count],
        color=['Fake News', 'Real News'],
        color_discrete_map={'Fake News': '#ef4444', 'Real News': '#22c55e'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Prepare display dataframe
    display_df = pd.DataFrame({
        'Text': [r['text'][:150] + '...' if len(r['text']) > 150 else r['text'] for r in results],
        'Classification': results_df['prediction']
    })
    
    # Color code the predictions
    def color_predictions(row):
        if row['Classification'] == 'Fake':
            return ['background-color: #fee2e2', 'background-color: #fecaca']
        else:
            return ['background-color: #dcfce7', 'background-color: #bbf7d0']
    
    st.dataframe(
        display_df.style.apply(color_predictions, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"fake_news_detection_results_{len(results)}_articles.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()