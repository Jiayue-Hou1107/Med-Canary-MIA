# Memorization Phase Transitions Expose Privacy Risks in Fine-Tuned Medical Large Language Models

This is the official implementation of the paper: **"Memorization Phase Transitions Expose Privacy Risks in Fine-Tuned Medical Large Language Models"**.

------

## Project Overview

The project is divided into four main phases:

1. **Data Synthesis & Processing**: Enhancing MIMIC-IV clinical notes with synthetic PII for privacy research.
2. **LoRA Fine-tuning**: Training multiple LLMs (Llama 2, Llama 3.1, Qwen 2.5/3) on clinical summarization tasks.
3. **Dynamic Parameter MIA**: Implementing Membership Inference Attacks with medical-weighted scoring.
4. **Memorization Analysis**: Calculating Exposure metrics for canary and high-frequency tokens to evaluate model memory.

## Directory Structure

Plaintext

```
├── clin-bhc-summ-main/        
│   ├── clin-bhc-summ/ 		      # Place your raw datasets and model weights here
│   ├── src/                      # Core training and inference scripts
│   │   ├── train_peft.py         # Fine-tuning script using LoRA
│   │   ├── run_peft.py           # Inference and testing script
│   │   └── constants.py          # Global configurations and hyperparameters
├── memory_analysis/              # Privacy analysis and MIA implementation
│   │   ├── dynamic_weighted_mia_attack.ipynb       # Comprehensive MIA analysis
│   │   ├── canaries_exposure.py  # Canary word exposure calculation
└── └── └── high_freq_exposure.py # High-frequency word exposure calculation
```

------

## 1. Data Processing

We utilize clinical notes from the **MIMIC-IV v2.2 Notes** dataset. The focus is on extracting the mapping between clinical "Findings" and the "Brief Hospital Course" (BHC).

- **Bucket Strategy**: Samples are filtered by token length and divided into three equal-sized buckets to ensure length diversity.
- **Dataset Split**:
  - **Train**: 6,000 samples
  - **Validation**: 300 samples
  - **Test**: 900 samples
- **Privacy Enhancement**: To simulate real-world medical data and evaluate privacy risks, we augmented the de-identified MIMIC notes with synthetic **Quasi-Identifiers (Q-IDs)**, including:
  - Full Name, Phone Number, Social Security Number (SSN), Zip Code, Gender, and Age.
- **Note on Data Access**: Due to MIMIC-IV data use agreements, the dataset cannot be publicly shared. Users must obtain their own license through PhysioNet and place the processed data in the `clin-bhc-summ/` directory.

------

## 2. Model Fine-tuning

We perform Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** on the following models:

- **Llama 2 (13B)** / **Llama 3.1 (8B)**
- **Qwen 2.5 (14B)** / **Qwen 3 (8B)**

**Training Details**:

- **Task**: Clinical BHC Summarization.
- **Epochs**: 20 (Intentionally set high to observe overfitting behaviors).
- **Observations**: Validation loss typically reaches its minimum around **Epoch 8**.
- **Configuration**: All hyperparameters (rank, alpha, learning rate, etc.) are managed in `src/constants.py`.

------

## 3. Dynamic Parameter MIA Attack

Located in the `memory_analysis/` directory, this module implements a **Membership Inference Attack (MIA)** based on dynamic weighting.

- **Mechanism**: The attack analyzes the model's output probabilities by contrasting weights between **Medical Vocabulary** and **General Vocabulary**.
- **Customization**: Users can adjust the `DESTINY` (weighting factor) to observe how specialized medical knowledge affects the success rate of the attack.

------

## 4. Memorization Analysis (Exposure)

To quantitatively analyze model memory, we perform comparative experiments using **Exposure metrics**:

- **Canary Exposure**: Measured via `run_canaries_peft.py`. It tracks how deeply the model remembers rare "canary" tokens inserted into the BHC.
- **High-Frequency Exposure**: Measured via `high_freq_exposure.py`. It evaluates the model's bias towards common clinical patterns and phrases.

These metrics help in understanding the gap between general learning and rote memorization in clinical LLMs.