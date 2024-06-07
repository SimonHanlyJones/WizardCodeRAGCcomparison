import os
import json

from langchain.chains import LLMChain
from langchain.llms import PredictionGuard

import predictionguard as pg
from langchain.prompts import PromptTemplate
from langchain.document_loaders import JSONLoader

import lancedb
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer

from datasets import load_dataset
import pandas as pd

from fastapi import FastAPI, HTTPException
import uvicorn

# De-provisioned PredictionGuard API key
os.environ["PREDICTIONGUARD_TOKEN"] = "xQZQCxe46UmYwYhjE6C0anJXec8bccK2mOXzwyvj"  # required for chatbot requests

import re

class ChatbotAPI:
    def __init__(self):
        self.app = FastAPI()
        self.chatbot_no_rag = Chatbot_no_RAG()
        self.rag_database = RAGDatabase()
        self.chatbot_with_rag = RAGChatbot(self.rag_database)

        @self.app.get(
            "/get_code_help/", description="Get help from our ai code assistant!"
        )
        def get_code_help(query: str):
            try:
                answer = self.chatbot_no_rag.query_model(query)
                return {"answer": answer}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get(
            "/get_code_help_with_rag/",
            description="Get help from our ai code assistant with RAG injected python documentation!",
        )
        def get_code_help_with_rag(query: str):
            try:
                answer = self.chatbot_with_rag.query_model(query)
                return {"answer": answer}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=8000)



class Chatbot_no_RAG:
    def __init__(self):
        code_prompt_template = (
            "### Instruction:\n"
            "You are a python code explanation assistant. Respond with a detailed explanation of the code snippet in the below input.\n"
            "\n"
            "### Input:\n"
            "{query}\n"
            "\n"
            "### Response:\n"
        )
        code_prompt = PromptTemplate(
            template=code_prompt_template, input_variables=["query"]
        )
        self.code_model = LLMChain(
            prompt=code_prompt, llm=PredictionGuard(model="WizardCoder"), verbose=False
        )

    def query_model(self, query):
        answer = self.code_model.predict(query=query, with_context=False)
        return answer


class RAGChatbot:
    def __init__(self, rag_database):
        self.rag_database = rag_database

        code_prompt_template = (
            "### Instruction:\n"
            "You are a python code explanation assistant. Respond with a detailed explanation of the code snippet in the below input. Additional python documentation is to be used only if you do not understand the code snippet.\n"
            "\n"
            "### Input:\n"
            "{query}\n"
            "Python Documentation: {rag_context}\n"
            "\n"
            "### Response:\n"
        )

        code_prompt = PromptTemplate(
            template=code_prompt_template, input_variables=["query", "rag_context"]
        )

        self.code_model = LLMChain(
            prompt=code_prompt, llm=PredictionGuard(model="WizardCoder"), verbose=False
        )

    def query_model(self, query):
        rag_context = self.rag_database.get_context_for_query(query)
        print("RETRIEVED:", rag_context[:90], "...")
        answer = self.code_model.predict(query=query, rag_context=rag_context,
                                          with_context=False
                                         )
        answer = answer.replace('\n\n\n\n\n\n', '\n\n')
        answer = answer.replace('\n\n\n\n\n', '\n\n')
        answer = answer.replace('\n\n\n\n', '\n\n')
        answer = answer.replace('\n\n\n', '\n\n')
        answer = self.truncate_to_complete_sentence(answer)
        return answer
    
    def truncate_to_complete_sentence(self, answer):
        full_stop_index = 0
        for i, char in enumerate(answer):
            if char == "." or char == "?" or char == "!":
                full_stop_index = i

        answer = answer[: full_stop_index + 1]
        return answer


class RAGDatabase:
    def __init__(self):
        name = "all-MiniLM-L12-v2"
        self.embedding_model = SentenceTransformer(name)

        database_path = ".lancedb" + name
        self.db, self.primary_table = self._initialise_lance_database(
            database_path)

    def _initialise_lance_database(self, database_path):
        db = lancedb.connect(database_path)
        
        table = db.open_table("linux")

        return db, table

    def get_context_for_query(self, query):
        num_batches = 1

        def embed_query(query):
            return self.embedding_model.encode(query)

        results = (
            self.primary_table.search(embed_query(query)).limit(num_batches).to_pandas()
        )
        results.sort_values(by=["_distance"], inplace=True, ascending=True)

        ret_context = results["text"].values[:num_batches]

        return ret_context[0]


def test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query):

    print("\n\n\nQUESTION:\n", query ,"\n")
    answer = chatbot_no_rag.query_model(query)
    print("ANSWER WITH NO RAG:\n", answer, "\n")

    answer = chatbot_with_rag.query_model(query)
    print("\nANSWER WITH RAG:\n", answer, "\n")



def demo(chatbot_no_rag, chatbot_with_rag):
    
    query = "print('Hello party animals!!!!!')"
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = "x = max(2, 3)"
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = "if(x > 0):\n    print('x is positive')\nelse:\n    print('x is negative')"
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = "x = 42\n" "for i in range(x):\n\n" "    print('Hello, World!')\n"
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = """
        dicts_lists = [
            {
                "Name": "James",
                "Age": 20,
            },
            {
                "Name": "May",
                "Age": 14,
            },
            {
                "Name": "Katy",
                "Age": 23,
            }
        ]
        dicts_lists.sort(key=lambda item: item.get("Age"))
    """
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = """
        a = ['blue', 'green', 'orange', 'purple', 'yellow']
        b = [3, 2, 5, 4, 1]

        sortedList =  [val for (_, val) in sorted(zip(b, a), key=lambda x: \
                x[0])]
    """
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = """
        #Formatting strings with f string.
        str_val = 'books'
        num_val = 15
        print(f'{num_val} {str_val}') # 15 books
        print(f'{num_val % 2 = }') # 1
        print(f'{str_val!r}') # books

        #Dealing with floats
        price_val = 5.18362
        print(f'{price_val:.2f}') # 5.18

        #Formatting dates
        from datetime import datetime;
        date_val = datetime.utcnow()
        print(f'{date_val=:%Y-%m-%d}') # date_val=2021-09-24
    """
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)

    query = (
        "def compute():\n"
        "    PERIMETER = 1000\n"
        "    for a in range(1, PERIMETER + 1):\n"
        "        for b in range(a + 1, PERIMETER + 1):\n"
        "            c = PERIMETER - a - b\n"
        "            if a * a + b * b == c * c:\n"
        "                # It is now implied that b < c, because we have a > 0\n"
        "                return str(a * b * c)"
    )
    test_case_sd_out(chatbot_with_rag, chatbot_no_rag, query)


if __name__ == "__main__":
    chatbot_api = ChatbotAPI()

    

    print("\n\nDEMO:\n")
    demo(chatbot_api.chatbot_no_rag, chatbot_api.chatbot_with_rag)

    chatbot_api.run()
