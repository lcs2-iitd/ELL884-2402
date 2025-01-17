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
