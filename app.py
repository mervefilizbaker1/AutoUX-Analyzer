import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import spacy
import time
import nltk
from nltk import bigrams
from transformers import pipeline

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def preprocessing(text):
    if pd.isna(text): return ""
    doc = nlp(str(text).lower())
    
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
    return " ".join(tokens)

def final_nlp_dashboard(file_obj, model_choice, progress=gr.Progress()):
    try:
        if file_obj is None: 
            return None, None, None, "<div style='color:red;'>Please upload a CSV file.</div>"
        
        progress(0.1, desc="Reading file...")
        df = pd.read_csv(file_obj.name)
        df_subset = df.head(500).copy() 
        
        start_time = time.time()

        # Smart Column Finder
        all_cols = df_subset.columns.tolist()
        summary_col = next((c for c in all_cols if 'summary' in c.lower() or 'desc' in c.lower()), all_cols[0])
        comp_col = next((c for c in all_cols if 'component' in c.lower() or 'part' in c.lower()), None)
        
        if not comp_col:
            comp_col = all_cols[1] if len(all_cols) > 1 else all_cols[0]
        
        progress(0.3, desc="Preprocessing text...")
        df_subset['cleaned_text'] = df_subset[summary_col].apply(preprocessing)
        
        # Custom Stopwords
        stpwords = ['car', 'vehicle', 'driver', 'mile', 'contact', 'tl', 'drive', 'failure','issue','problem','2016','take','not','happen','come','go',
                        'cause','new','get','work','use','make','see','even','also','would','could','like','one','two','three','own','state','dealer','dealership',
                          'time', 'manufacturer', 'notify', 'service','repair','approximate', 'mileage','tell','ford','say','recall','day','honda']
        
        all_words = " ".join(df_subset['cleaned_text']).split()
        filtered_words = [w for w in all_words if w not in stpwords]
        
        #Analysis
        if model_choice == "Keyword-based (Baseline)":
            counts = Counter(filtered_words).most_common(15)
            status = "Analysis: Single Keyword Frequency (Baseline)"
            raw_sample = df_subset[[summary_col, comp_col]].sample(min(10, len(df_subset)))
            
        elif model_choice == "N-Gram Analysis":
            progress(0.6, desc="Computing N-Grams...")
            bg_list = list(bigrams(filtered_words))
            bg_formatted = [" ".join(bg) for bg in bg_list]
            counts = Counter(bg_formatted).most_common(15)
            status = "Analysis: Contextual Word-Pair (Bigram) Patterns"
            raw_sample = df_subset[[summary_col, comp_col]].sample(min(10, len(df_subset)))
            
        else: # Transformer (Neural)
            progress(0.6, desc="Neural Sentiment Inference...")
            sample_df = df_subset.sample(min(10, len(df_subset))).copy()
            texts = sample_df[summary_col].tolist()
            neural_results = sentiment_pipeline(texts)
            
            sample_df['Sentiment'] = [res['label'] for res in neural_results]
            sample_df['Confidence'] = [f"%{res['score']*100:.1f}" for res in neural_results]
            
            neg_count = sum(1 for x in neural_results if x['label'] == 'NEGATIVE')
            avg_conf = sum(res['score'] for res in neural_results) / len(neural_results)
            
            status = f"Neural Analysis: {neg_count} high-priority issues | Avg Confidence: %{avg_conf*100:.1f}"
            raw_sample = sample_df[[summary_col, comp_col, 'Sentiment', 'Confidence']]
            counts = Counter(filtered_words).most_common(15)

        # Visualization
        progress(0.8, desc="Generating Visuals...")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[c[1] for c in counts], y=[c[0] for c in counts], palette='viridis', ax=ax_bar)
        ax_bar.set_title(f"Top 15 Insights - {model_choice}", fontsize=14)
        plt.tight_layout()
        
        wc_text = " ".join(filtered_words) if filtered_words else "no data"
        wc = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(wc_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
        
        # D. Dashboard 
        inf_time = f"{time.time() - start_time:.2f} sec"
        severity_html = f"""
        <div style='background-color: white; color: black; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #ddd; font-family: sans-serif;'>
            <h3 style='margin:0; font-weight: normal;'>Strategy: {model_choice}</h3>
            <p style='margin:8px 0; font-size: 0.9em; color: #555;'>{status}</p>
            <p style='margin:0; font-size: 1.1em;'>Processing Speed: <span style='color: red;'>{inf_time}</span></p>
        </div>
        """
        return fig_bar, fig_wc, raw_sample, severity_html
    
    except Exception as e:
        return None, None, None, f"<div style='color:red;'>An Error Occurred: {str(e)}</div>"

#GRADIO UI 
with gr.Blocks(theme=gr.themes.Soft(), title="AutoUX NLP Dashboard") as demo:
    gr.Markdown("# 🚗 AutoUX: Intelligent Vehicle Safety Analyzer")
    gr.Markdown("Merve Filiz Baker NLP Final Project")
    
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            file_input = gr.File(label="1. Upload NHTSA Complaints (CSV)")
            model_input = gr.Radio(
                choices=["Keyword-based (Baseline)", "N-Gram Analysis", "Transformer (Neural)"],
                label="2. Choose NLP Strategy",
                value="Keyword-based (Baseline)"
            )
            run_btn = gr.Button("🚀 Start Real-Time Analysis", variant="primary")
        
        with gr.Column(scale=2):
            severity_display = gr.HTML("<div style='text-align:center; padding:40px; border:1px dashed #ccc; border-radius:10px;'>Upload a file and select a strategy to begin.</div>")

    with gr.Tabs():
        with gr.TabItem("📊 Analysis Insights"):
            with gr.Row():
                plot_out = gr.Plot(label="Issue Frequency")
                cloud_out = gr.Plot(label="Theme Cloud")
        
        with gr.TabItem("🔍 Interactive Data"):
            gr.Markdown("### Data Explorer (Dynamic Analysis)")
            table_out = gr.DataFrame(interactive=True)
            
        with gr.TabItem("🔬 Model Insights"):
            gr.Markdown("""
            ### NLP Methodology Comparison
            * **Keyword-based:** Simple frequency distribution. Best for high-level word clouds.
            * **N-Gram Analysis:** Captures word pairs. Essential for technical terms like *'brake failure'*.
            * **Transformer (Neural):** Uses **DistilBERT** to understand semantic meaning and sentiment severity.
            """)

    run_btn.click(
        fn=final_nlp_dashboard,
        inputs=[file_input, model_input],
        outputs=[plot_out, cloud_out, table_out, severity_display]
    )

if __name__ == "__main__":
    demo.launch(share=True)