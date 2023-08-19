import os
import re
import pandas as pd
import datasets
import torch
from datetime import datetime, timedelta
import argparse
import questionary
torch.set_grad_enabled(False)
from InstructorEmbedding import INSTRUCTOR

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'instructor.bin')
DS_PATH = os.path.join(BASE_DIR, 'ds.bin')
FAISS_PATH = os.path.join(BASE_DIR, 'faiss.bin')
LOG_PATH = os.path.join(BASE_DIR, 'log.md')
COLUMN = 'embedding'
PROMPT = "Represent the Stackoverflow question for retrieving corresponding codes: "

'''Markdown->Df'''
def md2df(markdown_content):
    code_blocks = re.findall(r'```.*?```', markdown_content, flags=re.DOTALL)
    
    for i, block in enumerate(code_blocks):
        markdown_content = markdown_content.replace(block, f"CODE_BLOCK_{i}", 1)
    
    sections = re.split(r'(^#+ .*)', markdown_content, flags=re.MULTILINE)
    sections = [section.strip() for section in sections if section.strip()]
    
    for i, block in enumerate(code_blocks):
        sections = [section.replace(f"CODE_BLOCK_{i}", block) for section in sections]
    
    headings = sections[::2]
    contents = sections[1::2]

    df = pd.DataFrame({
        'Heading': headings,
        'Content': contents
    })

    return df

class VectorMD:
    def __init__(self, markdown_file=None, use_disk=False):
        self._model = None
        self._ds = None
        
        if markdown_file:
            self.setup(markdown_file, use_disk)

    @property
    def model(self):
        if self._model is None:
            if os.path.exists(MODEL_PATH):
                self._model = torch.load(MODEL_PATH)
            else:
                raise RuntimeError("Model not initialized. Please run the setup first.")
        return self._model

    @property
    def ds(self):
        if self._ds is None:
            if os.path.exists(DS_PATH):
                self._ds = datasets.load_from_disk(DS_PATH)
                self._ds.load_faiss_index(COLUMN, file=FAISS_PATH)
            else:
                raise RuntimeError("Dataset not initialized. Please run the setup first.")
        return self._ds

    def setup(self, markdown_file, use_disk=False):
        with open(markdown_file) as f:
            content = f.read()
        df = md2df(content)
        df['heading_trimmed'] = df['Heading'].str.replace(r'^#+\s', '', regex=True)
        ds = datasets.Dataset.from_pandas(df)

        model = INSTRUCTOR('hkunlp/instructor-large')
        ds = ds.map(lambda example: {COLUMN: model.encode([[PROMPT, instr] for instr in example['heading_trimmed']])}, batched=True, batch_size=8)
        ds = ds.remove_columns('heading_trimmed')
        if use_disk:
            torch.save(model, MODEL_PATH)
            ds.save_to_disk(DS_PATH)
            ds.add_faiss_index(COLUMN)
            ds.save_faiss_index(COLUMN, file=FAISS_PATH)
        else:
            ds.add_faiss_index(COLUMN)
            self._model = model
            self._ds = ds

    def query(self, query, use_questionary=False):
        question_embedding = self.model.encode([[PROMPT, query]])
        scores, results = self.ds.get_nearest_examples(COLUMN, question_embedding, k=5)
        if use_questionary == True:
            choices = results['Heading']
            choice_to_index = {choice: idx for idx, choice in enumerate(choices)}
            selected_choices = questionary.checkbox("Select results:", choices=choices).ask()
            selected_indexes = [choice_to_index[choice] for choice in selected_choices]
            str_results = '\n\n'.join([f'#{results["Heading"][i]} ({scores[i]})\n\n{results["Content"][i]}' for i in selected_indexes])
            utc_timestamp = datetime.utcnow() + timedelta(hours=9)
            str_formatted = f'\n\n# {query} ({utc_timestamp})\n\n{str_results}'
            with open(LOG_PATH, "a", encoding="utf-8") as text_file:
                text_file.write(str_formatted)
            return str_formatted
        else:
            return pd.DataFrame(list(zip(results['Heading'], results['Content'], scores)), columns = ["Heading", "Content", "Score"])
    
def setup_cli():
    parser = argparse.ArgumentParser(description='Initialize VectorMD')
    parser.add_argument('--file', type=str, default='vecDB.md', help='Path to the markdown file')
    args = parser.parse_args()

    if os.path.exists(args.file):
        VectorMD().setup(args.file, True)
    else:
        print(f"File {args.file} not found.")
        file_path = input("Please provide the path to your markdown file: ")
        VectorMD().setup(file_path, True)

def query_cli():
    parser = argparse.ArgumentParser(description='Search')
    parser.add_argument('user_input', type=str, nargs='*', default=[], help='User input')
    args = parser.parse_args()
    
    vmd = VectorMD()
    if args.user_input:
        user_query = ' '.join(args.user_input)
        print(vmd.query(user_query, True))
    else:
        while True:
            user_query = input("\nQuery (or type 'quit' to exit): ")
            if user_query.strip().lower() == "quit":
                break
            print(vmd.query(user_query, True))
            
if __name__ == "__main__":
    pass