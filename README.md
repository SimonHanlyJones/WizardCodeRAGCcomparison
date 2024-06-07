# Intel & Predicition Guard Hackathon Challenge 5 Winning Project: Python Code Explanation Assistant

## Introduction

Welcome to our hackathon-winning project! This repository contains the code for a Python code explanation assistant, developed by Simon Hanly-Jones, Ryan Mann, and Emmanuel Isacc, as part of the Prediction Guard and Intel hackathon. Our project leverages sequence transformers and Retrieval-Augmented Generation (RAG) to provide detailed explanations of Python code snippets.

Note that the resources made available during the hackathon have been de-provisioned and the API used as the foundation of the model is no longer authorised.

## Overview

We aimed to create a Python code explanation assistant that provides accurate and detailed explanations of code snippets. Our approach involved two main models: one without RAG and one with RAG injection. By scraping official Python documentation and using the "all-MiniLM-L12-v2" sequence transformer, we enhanced the assistant's ability to retrieve relevant documentation when needed.

## Models

### Non-RAG Model

Our non-RAG model uses the following prompt template:

```python
code_prompt_template = (
    "### Instruction:\\n"
    "You are a python code explanation assistant. Respond with a detailed explanation of the code snippet in the below input.\\n"
    "\\n"
    "### Input:\\n"
    "{query}\\n"
    "\\n"
    "### Response:\\n"
)
```

### RAG Model

The RAG model uses this prompt template:

```python
code_prompt_template = (
    "### Instruction:\\n"
    "You are a python code explanation assistant. Respond with a detailed explanation of the code snippet in the below input. Additional python documentation is to be used only if you do not understand the code snippet. \\n"
    "\\n"
    "### Input:\\n"
    "{query}\\n"
    "Python Documentation: {rag_context}\\n"
    "\\n"
    "### Response:\\n"
)
```

## Key Findings

- Good prompting was crucial for performance. Referring to the RAG injection explicitly as "Python Documentation" helped obtain meaningful contributions from the RAG model.
- Using terms like “Context” or “Informational Context” confused the model, making it think the user was asking about the Python documentation instead of the code snippet.

## Project Structure

Our code is split into two main files:

1. `get_lance_db.py`: Creates the database for RAG by transforming the `pychunks.json` file into a vectorized database.
2. `wiz_code.py`: Contains the chatbots and the FastAPI interface, and generates demo output to stdout before launching the FastAPI.

## Running the Project

To run the project, follow these steps:

1. Download all files and place them in the same directory.
2. Run `get_lance_db.py` to create the vectorized database.
3. Run `wiz_code.py` to generate the demo output and launch the FastAPI.

The FastAPI will be available at `127.0.0.1:8000/docs`, where you can access the automatically generated Swagger documentation as the user interface.

## Demo Output

### Example 1

**Question:**

```python
print('Hello party animals!!!!!')
```

**Answer with No RAG:**

The code snippet is a simple print statement that prints the string "Hello party animals!!!!!" to the console. The exclamation marks are added to make the string more exciting and funny.

**Answer with RAG:**

The code snippet is a simple print statement that outputs the string "Hello party animals!!!!!" to the console. The Python documentation is not required for this explanation.

### Example 2

**Question:**

```python
x = max(2, 3)
```

**Answer with No RAG:**

The `max()` function returns the largest item in a given iterable (e.g., a list, tuple, set, etc.). In this case, the iterable is `(2, 3)`. The `max()` function returns the largest item in this iterable, which is `3`. Therefore, the value of `x` is `3`.

**Answer with RAG:**

The code snippet is trying to find the maximum value of two numbers using the `max()` function. The `max()` function returns the largest item in an iterable or the largest of two or more arguments. In this case, the `max()` function is called with two arguments, 2 and 3. The `max()` function returns the largest of the two numbers, which is 3.

## Team

- [Simon Hanly-Jones](https://www.linkedin.com/in/simon-hanly-jones-79110572)
- [Ryan](https://www.linkedin.com/in/ryanlmann/)
- [Emmanuel Isaac](https://www.linkedin.com/in/emmanuel-isaac-b41882194)

We are available for DnD-based coding challenges and promise to try not to break all provided hardware.

---

We hope you find this project as exciting and informative as we did! Feel free to reach out to us for any questions or further collaboration opportunities.
