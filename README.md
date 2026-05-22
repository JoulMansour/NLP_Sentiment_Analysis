# 📈 NLP Domain Shift & Transfer Learning: Adapting DistilBERT to Financial Markets

*A portfolio project by Joul Mansour | Computer Science & Electrical Engineering*

## 🎯 Project Overview
This project explores the machine learning challenge of **Domain Shift**—how Natural Language Processing (NLP) models degrade when applied to out-of-distribution text—and implements modern **Parameter-Efficient Fine-Tuning (PEFT)** to solve it. 

The experiment takes a DistilBERT model trained entirely on subjective movie reviews (SST-2) and forces it to analyze highly formal, jargon-heavy financial news. I then utilized **LoRA (Low-Rank Adaptation)** to recover the model's accuracy on a limited compute budget, and conducted a secondary experiment using an LLM to generate synthetic training data.

## 🛠️ Technologies & Skills Demonstrated
* **Languages & Frameworks:** Python, PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`
* **AI/ML Techniques:** Transfer Learning, Zero-Shot Classification, LoRA / PEFT
* **Models Used:** `distilbert-base-uncased`, `SmolLM2-1.7B-Instruct`
* **Core Concepts:** Domain Shift, Synthetic Data Generation, AI Optimization

---

## 📊 Experimental Results

| Model State | Training Data | Evaluation Accuracy |
| :--- | :--- | :--- |
| **Baseline (SST-2)** | Movie Reviews | 91.06% |
| **Zero-Shot on Finance** | None (Domain Shift) | 21.00% |
| **Fine-Tuned with LoRA** | 500 Real Financial Texts | **74.77%** |
| **Synthetic LoRA (Bonus)** | 60 LLM-Generated Texts | 57.91% |

---

## 🔬 Methodology

### 1. The Baseline & The Distribution Shift
The project began with a DistilBERT model perfectly optimized for the SST-2 movie review dataset (91.06% accuracy). When tested on the `twitter-financial-news-sentiment` dataset, performance catastrophically dropped to **21.00%**. The model failed because it relied on emotional adjectives ("thrilling", "terrible") and lacked the architecture to predict the 3-class structure (Positive, Negative, Neutral) of formal financial jargon.

### 2. Parameter-Efficient Fine-Tuning (LoRA)
To fix the domain shift without retraining all 66 million parameters, I implemented LoRA. By freezing the DistilBERT backbone and attaching low-rank adapters specifically to the `q_lin` and `v_lin` attention modules, I successfully trained a new classification head on just 500 examples of human-labeled financial data. 
* **Outcome:** Accuracy rebounded to **74.77%**, proving that the model's core English understanding could be rapidly mapped to a new domain's vocabulary.

### 3. Synthetic Data Generation Experiment
To explore whether AI-generated data can substitute for human-labeled data, I deployed a local instance of `SmolLM2-1.7B-Instruct`. I prompted the LLM to generate 60 completely original financial headlines across the three sentiment classes.
* **Outcome:** A fresh model trained exclusively on this synthetic data achieved **57.91% accuracy**. While it significantly outperformed the 21% zero-shot baseline, it failed to beat the human-trained model, demonstrating that small-scale synthetic data lacks the linguistic diversity of real-world journalism needed for robust classification.

## 🚀 How to Run
The entire experiment is contained within a single Jupyter Notebook. It is optimized to run on a standard Google Colab T4 GPU instance. 

1. Clone the repository.
2. Open `NLP_Sentiment_Analysis.ipynb` in Google Colab or a local Jupyter environment.
3. Install the required dependencies: `pip install transformers datasets evaluate peft torchao`
4. Run the cells sequentially to replicate the baseline, the LoRA training loop, and the synthetic data generation.
