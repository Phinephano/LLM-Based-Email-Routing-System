# LLM-Based Email Routing System

> A comparative study of transformer-based approaches for automatic customer support email routing.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Customer support organizations receive thousands of customer emails every day. Manually routing these emails to the correct department is time-consuming and prone to human error.

This project develops an intelligent email routing system using Large Language Models (LLMs) and transformer-based architectures. Three different approaches are implemented and compared:

- GPT-2 Prompting (Zero/Few-shot Baseline)
- GPT-2 with LoRA Fine-Tuning
- DistilBERT Fine-Tuning

The models classify customer emails into one of five support departments.

---

## Dataset

Dataset:

**Customer Support Tickets**

https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets

Original Dataset

- 61,765 customer support tickets
- 52 support departments
- English and German languages

Dataset after preprocessing

- English tickets only
- Five target departments
- 16,562 samples

Target Departments

- Technical Support
- Customer Service
- Billing and Payments
- Sales and Pre-Sales
- General Inquiry

---

## Project Workflow

Raw Dataset

↓

Exploratory Data Analysis (EDA)

↓

Dataset Preprocessing

↓

Agent 1 – GPT-2 Prompting

↓

Agent 2 – GPT-2 + LoRA

↓

Agent 3 – DistilBERT

↓

Performance Evaluation

↓

Model Comparison

---

## Repository Structure

```
.
├── notebook.ipynb
├── datapreparation.py
├── utils.py
├── figures/
├── results/
├── report/
├── README.md
└── requirements.txt
```

---

## Models

### Agent 1 – GPT-2 Prompting

A zero-shot prompting baseline using GPT-2 without supervised training.

---

### Agent 2 – GPT-2 + LoRA

Parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA).

---

### Agent 3 – DistilBERT

A supervised transformer classifier fine-tuned on the customer support dataset.

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| GPT-2 Prompting | 56.50% | 41.56% | 56.50% | 45.65% |
| GPT-2 + LoRA | 65.86% | 65.45% | 65.86% | 65.22% |
| DistilBERT | **72.17%** | **71.78%** | **72.17%** | **71.93%** |

DistilBERT achieved the highest overall performance, demonstrating the effectiveness of supervised transformer-based classification over prompt-based and parameter-efficient approaches.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- PEFT (LoRA)
- Scikit-learn
- Matplotlib
- Pandas

---

## Installation

```bash
git clone https://github.com/USERNAME/NLP-Email-Routing-Transformers.git

cd NLP-Email-Routing-Transformers

pip install -r requirements.txt
```

---

## Running the Project

Open

```
notebook.ipynb
```

and execute the notebook sequentially.

The notebook automatically:

- Downloads the dataset
- Preprocesses the data
- Trains the models
- Evaluates performance
- Generates figures and result files

---

## Author

**Finhas Demissie Bekele**

M.Sc. Artificial Intelligence

University of Verona

Academic Year 2025–2026

---

## License

MIT License
