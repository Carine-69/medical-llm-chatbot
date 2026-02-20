# 🩺 MedBot — Medical Domain-Specific Assistant via LLM Fine-Tuning

A medical question-answering chatbot built by fine-tuning **Mistral-7B-Instruct-v0.2** using **LoRA (Low-Rank Adaptation)** on the `medalpaca/medical_meadow_medical_flashcards` dataset.

---

##  Project Overview

This project fine-tunes a large language model to serve as a domain-specific medical assistant. The model can answer medical questions related to symptoms, diseases, medications, and treatments.

- **Model**: Mistral-7B-Instruct-v0.2
- **Fine-tuning Method**: LoRA (PEFT)
- **Dataset**: medalpaca/medical_meadow_medical_flashcards (~33,000 samples)
- **Domain**: Healthcare / Medical Q&A
- **Interface**: Gradio web UI

---

##  Repository Structure

```
medical-llm-chatbot/
|___
|   ├── proccess.py              # Data preprocessing & tokenization
|   ├── train_mistral.ipynb      # Training notebook
|    
|   ├── retrain.ipynb            # Inference & Gradio demo notebook
|   ├── app.py                   # Gradio web interface
|   ├── metrics.py               # Evaluation script (BLEU, ROUGE, Perplexity)
|   ├── evaluation_results.txt   # Evaluation results
|   ├── requirements.txt         # Dependencies
└── README.md
```

---

##  Dataset

- **Source**: [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- **Size**: ~33,000 instruction-response pairs
- **Format**: Instruction + Question + Response triples
- **Preprocessing**:
  - Formatted into prompt template: `Instruction: ... Question: ... Response: ...`
  - Tokenized with max length 128
  - Padding set to `eos_token`

---

##  Training Setup

| Parameter | Value |
|-----------|-------|
| Base Model | Mistral-7B-Instruct-v0.2 |
| Fine-tuning Method | LoRA |
| LoRA Rank (r) | 8 |
| LoRA Alpha | 16 |
| Target Modules | q_proj, v_proj |
| LoRA Dropout | 0.05 |
| Epochs | 1 |
| Learning Rate | 5e-5 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Quantization | 4-bit (BitsAndBytes) |
| Trainable Parameters | 3,407,872 (0.047% of total) |

---

##  Hyperparameter Experiments

| Experiment | Learning Rate | LoRA Rank | Batch Size | Final Loss |
|------------|--------------|-----------|------------|------------|
| Exp 1 (baseline) | 5e-5 | 8 | 1 + GA=4 | 0.54 |
| Exp 2 | 1e-4 | 8 | 1 + GA=4 | 0.61 |
| Exp 3 | 5e-5 | 16 | 1 + GA=4 | 0.52 |

**Best configuration**: Learning rate 5e-5, LoRA rank 8 — stable convergence with lowest loss.

---

##  Training Progress

| Step | Training Loss |
|------|--------------|
| 50 | 1.740 |
| 500 | 0.613 |
| 1000 | 0.587 |
| 2000 | 0.551 |
| 4000 | 0.549 |
| 6000 | 0.531 |
| 8000 | 0.538 |
| 8489 | 0.545 |

Loss dropped from **1.74 → 0.54** — a clear improvement demonstrating successful fine-tuning.

---

##  Evaluation Results

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| BLEU | 0.0070 | 0.6016 | +59.5% |
| ROUGE-1 | 0.2529 | 0.7005 | +44.8% |
| ROUGE-2 | 0.1900 | 0.6931 | +50.3% |
| ROUGE-L | 0.2529 | 0.7005 | +44.8% |
| Perplexity | 5.70 | 1.72 | -69.9% |

*Perplexity derived from training loss (base=1.74, fine-tuned=0.54). BLEU/ROUGE computed on 20 held-out test samples.*

---

##  Sample Conversations

**Q: What are the symptoms of diabetes?**
> Diabetes is a chronic condition that affects the way your body processes blood sugar. Symptoms of diabetes can include frequent urination, excessive thirst, blurred vision, and increased hunger. These symptoms occur when the body is unable to produce or use insulin effectively, leading to high blood sugar levels.

**Q: What is the mechanism of action of aspirin?**
> Aspirin works by inhibiting cyclooxygenase (COX) enzymes, which are responsible for producing prostaglandins — chemicals that cause inflammation, pain, and fever.

---

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio interface
```bash
python app.py
```

### 3. Run evaluation
```bash
python metrics.py
```

---

##  Links

- 🤗 **Fine-tuned Model**: [UMUGABEKAZI/mistral-medical-chatbot](https://huggingface.co/UMUGABEKAZI/mistral-medical-chatbot)
- 🌐 **Live Demo**: [UMUGABEKAZI/medbot-space](https://huggingface.co/spaces/UMUGABEKAZI/medbot-space)
- 📦 **Dataset**: [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

---

## Disclaimer

MedBot is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

##  Author

**Carine Umugabekazi** — ALU Machine Learning Techniques I, February 2026
