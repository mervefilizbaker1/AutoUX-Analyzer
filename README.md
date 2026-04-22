---
title: AutoUX Analyzer
emoji: 🏆
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---
**AutoUX: Intelligent Vehicle Safety Analyzer**

NLP Final Project - User Guide & Technical Documentation

Live Demo: [(https://huggingface.co/spaces/Mfbaker/AutoUX-Analyzer)]

---

**Introduction**

AutoUX is an intelligent analytical dashboard designed to transform raw, unstructured vehicle safety data into actionable engineering insights. Every year, thousands of drivers report safety issues to the National Highway Traffic Safety Administration (NHTSA). However, for automotive engineers and UX researchers, manually reading these reports to find recurring patterns is time-consuming and prone to human error.

This system bridges that gap by using Natural Language Processing. At its core, AutoUX acts as a "smart filter" that automatically scans thousands of customer complaints, identifies the most frequently mentioned mechanical parts, and detects the emotional severity of each report.

---

**Usage**

The application is hosted on Hugging Face Spaces and designed for "one-click" execution.

**Launch:** Open the demo link provided above.

**Upload:** Upload a CSV file (e.g., NHTSA vehicle complaints). The system automatically detects relevant columns.

**Analyze:** Select an NLP Strategy (Keyword, N-Gram, or Transformer) and click "Start Real-Time Analysis".

**Interpret:** Use the Analysis Insights tab for frequency and theme visualizations.

Use the Interactive Data tab to see specific "Confidence" and "Severity" scores for each complaint (available in Transformer mode).

---

**Documentation (Behind the Scenes)**

***System Architecture & Process Flow***

The application architecture is divided into four main stages:

* **Data Ingestion:** Reads CSV input and applies a Smart Column Mapping logic to identify 'summary' and 'component' fields regardless of exact naming.

* **Preprocessing Pipeline:** Powered by spaCy (en_core_web_sm). It performs:

  * Tokenization & Lemmatization: Reducing words to their roots.

  * Stopword Removal: Filtering general English noise and domain-specific terms (e.g., "car", "vehicle").

* **Modeling Engine:** Routes cleaned text to the selected NLP architecture.

* **Visualization Layer:** Uses Seaborn and WordCloud to render mathematical counts into human-readable insights.

***Model Details***

* **Keyword-based:** A baseline frequency distribution model using Python's collections.Counter.

* **N-Gram Analysis:** Uses NLTK to generate Bigrams. This captures context-dependent phrases (e.g., "brake" + "pedal").

* **Transformer (DistilBERT):**

  * Architecture: A distilled version of BERT, featuring 6 layers, 768 hidden dimensions, and 12 attention heads.

  * Training: Pre-trained on the BooksCorpus and English Wikipedia; fine-tuned on the SST-2 dataset for sentiment classification.

  * Compute: Optimized for CPU inference (approx. 512MB RAM requirement).

  * Biases/Limitations: The model may struggle with highly technical sarcasm or rare automotive jargon not present in its training corpus.

***Dataset Description***

The system is optimized for NHTSA Complaint Data.

* **Sampling:** For the demo, the system samples the first 500-5000 rows to ensure low latency.

* **Annotation:** The data consists of unstructured user summaries and structured "Component" labels provided by the NHTSA.

* **Filtering:** Rows with missing summaries are automatically handled to prevent pipeline breaks.


***External Frameworks & Library Functionality***

The system is built using a professional stack of Python libraries:

* **Gradio:** Manages the web-based UI and handles the asynchronous communication between the server and the user.

* **Pandas:** Efficiently processes the 224.5 MB dataset, managing thousands of rows for real-time sampling and analysis.

* **Transformers (Hugging Face):** Provides the pipeline API for the DistilBERT model, enabling deep semantic analysis of complaint summaries.

* **spaCy (en_core_web_sm):** Powers the text preprocessing stage with professional-grade lemmatization and tokenization.

* **Matplotlib & Seaborn:** Generate the visual analytics, converting complex frequency distributions into easy-to-read dashboard charts.

***Experiments and Comparative Analysis***

I conducted a comparative experiment using a large-scale NHTSA dataset (224.5 MB) to evaluate the three NLP strategies. The objective was to measure the trade-off between speed (latency) and the depth of insight provided.

| Strategy               | Processing Time | Insight Type        | Qualitative Observation |
|----------------------|----------------|--------------------|-------------------------|
| Keyword (Baseline)   | 11.18 sec      | Frequency          | Fast, but lacks context. Highlights single words like "brake". |
| N-Gram Analysis      | 10.25 sec      | Pattern Recognition| Most Efficient. Captures phrases like "steering wheel" and "air bag". |
| Transformer (Neural) | 11.73 sec      | Semantic Severity  | Highest Depth. Identified 10/10 issues as High Priority with 99.8% Avg. Confidence. |

***Key Findings & Analysis:***

* **The Speed-to-Intelligence Ratio:** Surprisingly, the Transformer model (11.73s) performed nearly as fast as the Keyword baseline (11.18s) despite its deep learning architecture. This proves that DistilBERT is an excellent choice for real-time automotive safety applications.

* **Contextual Superiority:** While the Baseline showed "brake" and "stop" separately, the N-Gram model successfully linked them into technical concepts. However, the Transformer model added a layer of "Emotional Intelligence" by identifying that 100% of the tested samples were high-priority/critical, reaching a near-perfect confidence score of %99.8.

* **Final Selection:** Based on these results, the N-Gram model is recommended for general technical audits, while the Transformer model is essential for "Emergency Filtering" where identifying the severity of a complaint is more important than just counting words.

---

**Contributions**

In building AutoUX, I implemented several unique features that differentiate it from basic NLP scripts:

* **Hybrid Comparative UI:** I created a single interface that allows users to swap between 3 generations of NLP models instantly.

* **Domain-Specific Filtering:** I developed a custom dictionary to filter out noise specific to the automotive industry, ensuring that "engine" or "brakes" stand out instead of "car" or "driver."

* **Dynamic Severity Scoring:** I integrated a live Transformer pipeline that adds a "Confidence" column to the raw data, allowing for data-driven prioritization.

---

**Limitations**

* **Language:** The preprocessing and Transformer models are currently limited to English.

* **Hardware Constraints:** On the free CPU tier, analyzing more than 10,000 rows simultaneously may cause a timeout.

* **Short Descriptions:** The Transformer model's accuracy decreases significantly with summaries shorter than 3 words (e.g., "it broke").

---

_Developed by Merve Filiz Baker for ARI 525 - University of Michigan-Flint._
