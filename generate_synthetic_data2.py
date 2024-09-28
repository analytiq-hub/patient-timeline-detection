#!/usr/bin/env python3

import pandas as pd
from langchain_ai21 import AI21LLM
from langchain_core.prompts import PromptTemplate
import OpenSSL
import os
import requests
import json
# Set your AI21 API key
import langchain_ai21 as ai21
# Read the API key from the environment variable
ai21.api_key = os.getenv("AI21LABS_API_KEY")
ai21_api_key = os.getenv("AI21LABS_API_KEY")
print(f"AI21Labs key: {ai21_api_key}")

def ai21_chat(message, system) -> str:
    url = "https://api.ai21.com/studio/v1/j2-ultra/chat"

    payload = {
        "numResults": 1,
        "temperature": 0.7,
        "messages": [
            {
                "text": message,
                "role": "user"
            }
        ],
        "system": system
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {ai21_api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)

    json_str = response.text
    # Convert the json string to a dict
    json_output = json.loads(json_str)
    return json_output["outputs"][0]["text"]

resp = ai21_chat(
    "I'm crafting a market analysis tool for fintech leaders. How should I initiate the process?",
    "You are an AI assistant for business research. Your responses should be informative and concise."
    )
print(resp)
exit(0)

# Get the __file__ directory
TOP = os.path.dirname(__file__)
INPUT_EVENTS = f"{TOP}/datasets/mimic/icu/inputevents.csv"

# Ensure the file is present
if not os.path.exists(INPUT_EVENTS):
    print(f"Input events file not found: {INPUT_EVENTS}")
    exit(1)

print(f"Reading input events from {INPUT_EVENTS}")

df = pd.read_csv(INPUT_EVENTS)

model = AI21LLM(model="j2-ultra", api_key=ai21_api_key)

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
    print(messages)
    print(ai_msg)
    return ai_msg

# Apply the function to the first 100 rows of the DataFrame
df_subset = df.head(100)
df_subset['clinical_summary'] = df_subset.apply(generate_clinical_summary, axis=1)

# Save the updated DataFrame to a new CSV file
df_subset.to_csv('/Users/Manas Goel/Desktop/agi/inputevents_with_summary.csv', index=False)
