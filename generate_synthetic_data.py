#!/usr/bin/env python3

import pandas as pd
from langchain_ai21 import AI21LLM
from langchain_core.prompts import PromptTemplate
import OpenSSL
import os

# Set your AI21 API key
import langchain_ai21 as ai21
# Read the API key from the environment variable
ai21.api_key = os.getenv("AI21LABS_API_KEY")
print(f"AI21Labs key: {ai21.api_key}")

# Get the __file__ directory
TOP = os.path.dirname(__file__)
INPUT_EVENTS = f"{TOP}/datasets/mimic/icu/inputevents.csv"

# Ensure the file is present
if not os.path.exists(INPUT_EVENTS):
    print(f"Input events file not found: {INPUT_EVENTS}")
    exit(1)

print(f"Reading input events from {INPUT_EVENTS}")

df = pd.read_csv(INPUT_EVENTS)

model = AI21LLM(model="j2-ultra", api_key='fdUeX7hXHjsB3J2YOxXIEgN6SnUnnydO')

def generate_clinical_summary(row):
    context = ""
    for col in df.columns:
        if pd.notna(row[col]):
            context += f"{col}: {row[col]}\n"

    template = """You are a helpful assistant that generates clinical notes using medical terminology. 
    Utilize reasoning to analyze the following structured data and provide a detailed, narrative clinical note 
    reflecting typical communication in clinical notes without suggesting solutions.
    
    Data:
    {context}
    """
    
    prompt = PromptTemplate.from_template(template)
    messages = prompt.format(context=context)

    ai_msg = model.invoke(messages)
    return ai_msg

# Apply the function to the first 100 rows of the DataFrame
df_subset = df.head(100)
df_subset['clinical_summary'] = df_subset.apply(generate_clinical_summary, axis=1)

# Save the updated DataFrame to a new CSV file
df_subset.to_csv('/Users/Manas Goel/Desktop/agi/inputevents_with_summary.csv', index=False)
