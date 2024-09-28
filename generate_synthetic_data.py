import pandas as pd
from langchain_ai21 import AI21LLM
from langchain_core.prompts import PromptTemplate

# Set your AI21 API key
import langchain_ai21 as ai21
ai21.api_key = 'fdUeX7hXHjsB3J2YOxXIEgN6SnUnnydO'  # Replace with your actual API key

df = pd.read_csv('/Users/Manas Goel/Desktop/agi/inputevents.csv')

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
