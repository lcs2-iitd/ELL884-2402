---
layout: page
title: Project
permalink: /project/
---

# Project Team Formation and Project Selection

Kindly fill out the [form](https://forms.office.com/r/QsXSWVFdxV) (Right click to open in new tab if the form does not open) before 30/01/2025 17:00 hours.

**Note:** Only one member per team should fill out the form. You need to fill in the top-5 preferences.

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

- Model: LLaMA-3.1-8B (pre-trained). 
- Role: Acts as the oracle, generating logits, embeddings, or task-specific outputs for training the student models. 

#### Student System 

- Constraints: Combined size ≤ 1.5B parameters. 
- Design Choices: 
  - Single Multi-Task Model: 
    - A unified student model trained for all tasks. 
  - Task-Specific Models: 
    - Separate smaller models specialized for each task. 
    - Shared encoder with task-specific decoders. 
  - Hybrid Approach: 
    - A shared backbone (e.g., Llama-3.2-1B, ~1B parameters) with task-specific adapters or lightweight modules. 
    - Use techniques like LoRA or prompt tuning. 
- Additional Guidelines: 
  - The student models shuold be pre-fine-tuned for any specific task. You can fine tune them using PEFT or FFT 
  - The student system must intelligently analyze input prompts and determine task-specific processing if using task-specific models. 

### Tasks and Datasets 

#### Tasks: 

1. Summarization: 
   - Dataset: CNN/DailyMail (news articles → abstractive summaries). 

2. Question Answering: 
   - Dataset: SQuAD 2.0 (context + question → answer or "no answer"). 

3. Paraphrase Generation: 
   - Dataset: Quora Question Pairs (questions → paraphrases). 

#### Dataset Usage: 

- Use only the train split for training. 
- The test split will be used for leaderboard evaluation. 

### Evaluation Metrics 

The quality of the generated outputs will be evaluated using the following metrics: 

1. Summarization: 
   - ROUGE-L. 

2. Question Answering: 
   - Combination of ROUGE-L and BERTScore. 

3. Paraphrase Generation: 
   - Combination of Sacre-BLEU and METEOR. 

4. Efficiency: 
   - Processing time per query (the standard hardware will be announced later). 

**Final Leaderboard Score**: A weighted combination of all the above metrics on the test datasets. Exceeding the 1.5B parameter constraint will result in exponential penalties. 

### Guidelines 

1. Teacher Model Constraints: Base pre-trained LLaMA-3.1-8B. 
2. Student Model Constraints: Open-source-trained LLMs are not fine-tuned for any specific task. 
3. Plagiarism Policy: 
   - Methods from published papers may be adapted, but the implementation must be original. 
   - Submissions will be checked for plagiarism against web resources and team submissions. Any detected cases will result in zero marks for the project component. 
4. Kaggle Competition: 
   - The project will be hosted as a Kaggle competition. 
5. Experimentation is the king. Try to experiment with as many techniques as you can. You might also need to resort to quantization and PEFT techniques for FT.
   - Ensure that your code runs smoothly in the Kaggle environment and generates output files that meet the competition specifications. Submission requirements may be subject to change. 

### Relevant Papers 

#### Knowledge Distillation 

1. Distilling the Knowledge in a Neural Network 
   - Authors: Geoffrey Hinton, Oriol Vinyals, Jeff Dean 
   - Link: [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531) 

2. A Survey on Knowledge Distillation of Large Language Models 
   - Link: [https://arxiv.org/abs/2402.13116](https://arxiv.org/abs/2402.13116) 

3. MiniLLM: Efficient Knowledge Distillation for Large Language Models 
   - Link: [https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543) 

#### Multi-Task Learning 

4. An Overview of Multi-Task Learning in Deep Neural Networks 
   - Authors: Sebastian Ruder 
   - Link: [https://arxiv.org/abs/1706.05098](https://arxiv.org/abs/1706.05098) 

#### Parameter-Efficient Fine-Tuning 

5. LoRA: Low-Rank Adaptation of Large Language Models 
   - Link: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) 

6. AdaLoRA: Adaptive Low-Rank Optimization for Efficient Fine-Tuning 
   - Link: [https://arxiv.org/abs/2303.10512](https://arxiv.org/abs/2303.10512) 

7. Prefix Tuning: Optimizing Continuous Prompts for Generation 
   - Link: [https://arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190) 

---

## Project 6: Building a Multi-Model System for Optimized Natural Language Generation 

### Problem Statement 

The goal of this project is to develop a multi-model system that leverages the strengths of different pre-trained models—Qwen2.5-1.5B, OPT-1.3B, and LLaMA-3.2 1B—to optimize performance across multiple tasks in Natural Language Generation (NLG). Unlike traditional single-model systems, this project focuses on combining multiple models in an intelligent and efficient way to balance accuracy, resource usage, and task-specific optimization. 

Students are encouraged to design systems that use innovative techniques, including but not limited to: 

- **Dynamic Decision Layers**: Decide which model(s) to query based on the input query or task type. 

- **Pipeline Architectures**: Use one model’s output as the input to another, creating a chain of processing for improved results. 

- **Ensemble Techniques**: Combine predictions from multiple models to produce a superior final output. 

The challenge lies in creating an efficient system that achieves high performance across tasks while minimizing redundancy and computational cost. 

### Tasks and Datasets 

The system will be evaluated on the following tasks and datasets: 

1. **Summarization**: 
   - Dataset: CNN/DailyMail (news articles → abstractive summaries). 
   - Task: Generate concise and informative summaries of news articles. 

2. **Question Answering**: 
   - Dataset: SQuAD 2.0 (context + question → answer or "no answer"). 
   - Task: Produce free-form answers based on a given context and question. 

3. **Paraphrase Generation**: 
   - Dataset: Quora Question Pairs (questions → paraphrases). 
   - Task: Generate semantically equivalent paraphrases for input sentences. 

You are only allowed to use the train split of these datasets for training purposes. The test split will be used for leaderboard evaluation. 

- [https://huggingface.co/datasets/cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) 
- [https://huggingface.co/datasets/squad](https://huggingface.co/datasets/squad) 
- [https://huggingface.co/datasets/quora](https://huggingface.co/datasets/quora) 

### Evaluation Metrics 

The quality of the generated outputs will be assessed using the following metrics: 

1. **Summarization**: ROUGE-L. 
2. **Question Answering**: Combination of ROUGE-L and BERTScore. 
3. **Paraphrase Generation**: Combination of Sacre-BLEU and METEOR. 
4. **Efficiency**: Inference time per query will be measured, and a standard hardware 
   specification will be announced later. 

The final leaderboard score will combine all these metrics, evaluated on the test splits of 
the specified datasets. 

### Guidelines 

1. **Model Constraints**: 
   - You are only allowed to use the following pre-trained language models: 
     - Qwen2.5-1.5B 
     - OPT-1.3B 
     - LLaMA-3.2 1B 

   - Fine-tuning on the train splits of the specified datasets is allowed. 

2. **Prohibited Models**: 
   - Publicly available models explicitly fine-tuned for these tasks are not allowed. 

3. **Plagiarism Policy**: 
   - Methods from published papers may be adapted, but the implementation must be original, with proper citations provided. 
   - Submissions will be checked for plagiarism against web resources and team submissions. Any detected cases of plagiarism will result in zero marks for the project component. 

4. **Kaggle Competition**: 
   - The project will be hosted as a Kaggle competition. 
   - Details regarding the competition, including submission requirements, will be shared soon. 
   - Ensure that your code runs smoothly in the Kaggle environment and generates output files that meet the competition specifications. Submission requirements may be subject to change. 

5. Experimentation is the king. Try to experiment with as many techniques as you can. You might also need to resort to quantization and PEFT techniques for FT 

### Relevant Papers 

To assist in designing your system, here are some relevant papers that provide insights into multi-model systems, ensemble techniques, and decision layers: 

#### Parameter Efficient Fine Tuning 

1. LoRA: Low-Rank Adaptation of Large Language Models 
   - Authors: Edward J. Hu, Yelong Shen, Phillip Wallis, et al. 
   - Link: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) 

2. Prefix-Tuning: Optimizing Continuous Prompts for Generation 
   - Authors: Xiang Lisa Li, Percy Liang 
   - Link: [https://arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190) 

#### Dynamic Decision Layers and Model Routing 

3. Mixture of Experts 
   - Authors: Noam Shazeer et al. 
   - Link: [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538) 

4. AdaBERT: Task-Adaptive BERT Compression with Mixture-of-Adapters 
   - Authors: Changlan Li et al. 
   - Link: [https://arxiv.org/abs/2005.04861](https://arxiv.org/abs/2005.04861) 

#### Ensemble and Modular Techniques 

5. Ensemble Methods in Machine Learning 
   - Authors: Thomas G. Dietterich 
   - Link: [https://link.springer.com/chapter/10.1007/3-540-45014-9_1](https://link.springer.com/chapter/10.1007/3-540-45014-9_1) 

6. RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks 
   - Authors: Patrick Lewis et al. 
   - Link: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) 

### Deliverables 

1. **Code**: 
   - A modular and scalable system integrating multiple models. 

2. **Report**: 
   - Description of the system architecture, design decisions, and task-specific performance. 

3. **Leaderboard Submission**: 
   - Outputs for the test splits, formatted as per the Kaggle competition requirements.
---

# Project 7: Taxonomy Expansion Using Prompt-Based Reasoning on Graphs

## Problem Statement

The goal of this project is to expand an existing taxonomy by accurately finding the parent node for a new concept. Instead of relying on discrimative methods, this project utilizes advanced prompt engineering and large language models (LLMs) to "think on graphs." The system will integrate semantic and structural reasoning over the taxonomy to identify the most suitable parent node.

## Framework Design

### Key Objectives

1. **Graph Representation**:
   - Represent the taxonomy as a structured text-based graph, where nodes and edges are described using natural language.
   - Include metadata such as node definitions, hierarchy levels, and relationships.

2. **Prompt-Based Parent Identification**:
   - Use structured prompts to reason about the graph and identify the best parent node.
   - Incorporate contextual information about the graph's structure and semantics into the prompts.

3. **Interactive Refinement**:
   - Iteratively refine the parent prediction by querying the model with progressively detailed prompts incorporating feedback and additional context.
   - You need to predict the parent node for the query term. For this, you may need to pass some possible parent nodes in the prompt to find the parent from them.

### Key Components

#### Graph Representation

- **Input Format**:
  - Convert the taxonomy graph into a textual representation (e.g., JSON, plain text).
  - Include:
    - Node names and descriptions.
    - Parent-child relationships.
    - Depth and sibling information (if available).

- **Example**:
  ```json
  {
    "node": "Mammal",
    "description": "Warm-blooded vertebrates with hair or fur.",
    "children": ["Dog", "Cat", "Whale"]
  }
  ```

  **Output**
  -	The selected parent node for the new concept.
 
  **Dataset**
  - SemEval Datasets: Science and Food
  - WordNet Dataset

**Evaluation Metrics**
- Accuracy
- Wu & Palmer Metric

**Guidelines**
- Use advanced LLMs (e.g., GPT-4, LLaMA) for prompt-based reasoning.
- Experiment with different prompt templates to improve accuracy and reasoning.
- Integrate pre-trained embedding models for semantic similarity computation.
- Ensure that the taxonomy remains a valid DAG after expansion.

**Relevant Papers and Resources**
- [Think on Graph](https://arxiv.org/abs/2307.07697)
- [TaxoExpan](https://arxiv.org/pdf/2001.09522)
- [HEF](https://arxiv.org/pdf/2101.11268)
- [QEN](https://dl.acm.org/doi/pdf/10.1145/3485447.3511943)
