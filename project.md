---
layout: page
title: Project
permalink: /project/
---

# Project Team Formation and Project Selection

Kindly fill out [the form](https://forms.office.com/r/wnFG7vw5ha) before 13/08/2024 11:59 PM.

**Note:** Only one member per team should fill out the form.

---

# List of Projects

## Project 1: Developing a System for Table QA

**Problem Statement**  
The aim of this project is to develop a system capable of answering complex, free-form questions based on tabular data. Unlike simple factual QA tasks, this project requires the system to understand the question, retrieve relevant information from the table, reason about the retrieved data, and generate a coherent and contextually accurate answer in natural language.

**Task Description**  
You are tasked with building a Table Question Answering (Table QA) system. The system must process a given table and a corresponding question to produce a free-form answer. The system should be able to:
- **Comprehend the Question:** Analyze and understand the intent of the question.
- **Retrieve Relevant Information:** Identify the necessary rows, columns, or cells in the table to answer the question.
- **Integrate and Infer:** Reason about the retrieved information, integrating multiple pieces of data when required.
- **Generate an Answer:** Produce a fluent and coherent free-form answer in natural language.

**Dataset**  
You will be using the FeTaQA dataset for this project. The provided HuggingFace version already has a train, val, and test split. Do not use the test split for training to ensure an unbiased evaluation. Final rankings will be based on performance on a private test dataset.

**Evaluation Metrics**  
The generated answers will be evaluated using:
- Sacre-BLEU (S-BLEU)
- ROUGE
- BERTScore

**Suggestions**  
- Develop a modular system with separate components for different sub-tasks instead of relying solely on a large language model.
- Explore techniques to optimize performance while minimizing trainable parameters.
- Innovative and efficient approaches will be rewarded during the presentation.

**Guidelines**  
- **Kaggle Competition:** The project will be hosted as a Kaggle competition. Details and submission requirements will be shared soon.
- **Model Constraints:** Use pre-trained language models with a maximum of 8 billion parameters. Avoid publicly available models fine-tuned for table-based tasks.
- **Plagiarism Policy:** Adaptations from published papers are allowed but must be implemented independently, with proper citation. Plagiarism will result in zero marks.

---

## Project 2: Building a Text-to-SQL System

**Problem Statement**  
The goal of this project is to design a system that translates natural language questions into SQL queries executable on a relational database. The system must handle complex queries involving joins, nested subqueries, and aggregation while generalizing to unseen databases.

**Task Description**  
You are required to develop a Text-to-SQL system capable of:
- **Comprehending Query Intent:** Parse natural language questions to determine intent and desired data output.
- **Understanding Database Schema:** Analyze the schema to identify relevant tables, columns, and relationships.
- **Generating SQL Queries:** Produce syntactically and semantically correct SQL queries.
- **Handling Complexity:** Address challenges involving joins, nested subqueries, and aggregation.

**Dataset**  
You will use the Spider dataset. Treat the provided val split as the test split and generate a separate val split from the train data. Final rankings will be based on a private test dataset.

**Evaluation Metrics**  
The system will be evaluated on:
- Component Matching
- Exact Matching  

**Suggestions**  
- Create a modular system with components for question parsing, schema understanding, and SQL generation.
- Consider using pre-trained language models while addressing their limitations in SQL generation.
- Innovative and efficient designs will be rewarded during the presentation.

**Guidelines**  
- **Kaggle Competition:** The project will be hosted on Kaggle, with details and submission requirements shared soon.
- **Model Constraints:** Use pre-trained models with a maximum of 8 billion parameters. Avoid models fine-tuned on text-to-SQL datasets.
- **Plagiarism Policy:** Adapt methods from published papers, but ensure independent implementation and citation. Plagiarism will result in zero marks.

---

## Project 3: DialoCONAN Counterspeech Generation Challenge

**Problem Statement**  
Your task is to develop an advanced counterspeech generation model using the DialoCONAN dataset. The DialoCONAN dataset comprises over 3000 multi-turn fictitious dialogues between a hater and an NGO operator, covering six targets of hate. Your goal is to generate high-quality, contextually appropriate counterspeech responses given a hate speech input and dialogue history.

**Dataset**  
The DialoCONAN dataset containing multi-turn dialogues will be provided. You can split the dataset into train, validation, and test sets for your experiments. The final test data released during the competition will be disjoint from the provided dataset.

**Evaluation Metrics**  
The generated counterspeech will be evaluated on:
- BLEU score
- ROUGE score
- BERTScore

**Kaggle Competition Guidelines**  
- All experiments must be conducted within the Kaggle environment.
- Participants are allowed to use pre-trained language models with up to 8 billion parameters.

---

## Project 4: IntentCONANv2 Intent-Specific Counterspeech Generation

**Problem Statement**  
Your challenge is to create an intent-specific counterspeech generation model using the IntentCONANv2 dataset. The IntentCONANv2 dataset contains around 13K counterspeeches conditioned on four intents (csType): informative, denouncing, question, and positive. Your objective is to generate high-quality, intent-specific counterspeech responses to given hate speech and csType as inputs.

**Dataset**  
The IntentCONANv2 dataset will be provided, containing hate speech-counterspeech pairs with associated intents. You can split the dataset into train, validation, and test sets for your experiments. The final test data released during the competition will be disjoint from the provided dataset.

**Evaluation Metrics**  
The generated counterspeech will be evaluated on:
- BLEU score
- ROUGE score
- BERTScore

**Kaggle Competition Guidelines**  
- All experiments must be conducted within the Kaggle environment.
- Participants are allowed to use pre-trained language models with up to 8 billion parameters.

---

## Project 5: Multi-Task Knowledge Distillation Framework for Natural Language Generation

### Problem Statement
The aim of this project is to develop a multi-task knowledge distillation system for Natural Language Generation (NLG). Unlike systems designed for a single task, this project requires the system to excel across multiple NLG tasks, such as summarization, question answering, and paraphrase generation. The focus is on distilling the knowledge of a large teacher model (LLaMA-7B) into a smaller, efficient system (≤1.5B parameters) that generalizes well across diverse tasks while maintaining high performance.

### Framework Design

#### Teacher Model
- **Model**: LLaMA-3.1-8B (pre-trained).
- **Role**: Acts as the oracle, generating logits, embeddings, or task-specific outputs for training the student models.

#### Student System
- **Constraints**: Combined size ≤ 1.5B parameters.
- **Design Choices**:
  - **Single Multi-Task Model**:
    - A unified student model trained for all tasks.
  - **Task-Specific Models**:
    - Separate smaller models specialized for each task.
    - Shared encoder with task-specific decoders.
  - **Hybrid Approach**:
    - A shared backbone (e.g., Llama-3.2-1B, ~1B parameters) with task-specific adapters or lightweight modules.
    - Use techniques like LoRA or prompt tuning.
- **Additional Guidelines**:
  - Student models should be pre-fine-tuned for any specific task using PEFT or FFT.
  - The student system must intelligently analyze input prompts and determine task-specific processing if using task-specific models.

### Tasks and Datasets
**Tasks**:
1. **Summarization**:
   - Dataset: CNN/DailyMail (news articles → abstractive summaries).
2. **Question Answering**:
   - Dataset: SQuAD 2.0 (context + question → answer or "no answer").
3. **Paraphrase Generation**:
   - Dataset: Quora Question Pairs (questions → paraphrases).

**Dataset Usage**:
- Use only the train split for training.
- The test split will be used for leaderboard evaluation.

### Evaluation Metrics
1. **Summarization**: ROUGE-L.
2. **Question Answering**: Combination of ROUGE-L and BERTScore.
3. **Paraphrase Generation**: Combination of Sacre-BLEU and METEOR.
4. **Efficiency**: Processing time per query (hardware specifications to be announced).

**Final Leaderboard Score**:
A weighted combination of all metrics on the test datasets. Exceeding the 1.5B parameter constraint results in exponential penalties.

### Guidelines
1. **Teacher Model Constraints**: Base pre-trained LLaMA-3.1-8B.
2. **Student Model Constraints**: Open source pre-trained LLMs not fine-tuned for any specific task.
3. **Plagiarism Policy**:
   - Adapt methods from published papers with original implementation.
   - Plagiarism checks against web resources and team submissions.
4. **Kaggle Competition**:
   - Hosted as a Kaggle competition.
   - Ensure smooth code execution and competition-specific output format.
   - 
---

## Project 6: Building a Multi-Model System for Optimized Natural Language Generation

**Problem Statement**  
Develop a multi-model system leveraging Qwen2.5-1.5B, OPT-1.3B, and LLaMA-3.2 1B to optimize NLG performance across summarization, question answering, and paraphrase generation.

**Tasks and Datasets**  
- Summarization: CNN/DailyMail  
- Question Answering: SQuAD 2.0  
- Paraphrase Generation: Quora Question Pairs  

**Evaluation Metrics**  
- Summarization: ROUGE-L  
- Question Answering: ROUGE-L and BERTScore  
- Paraphrase Generation: Sacre-BLEU and METEOR  

**Design Suggestions**  
- Dynamic Decision Layers  
- Pipeline Architectures  
- Ensemble Techniques  

---
