import os
import pandas as pd
from dotenv import load_dotenv
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat
from multiprocessing import Pool, cpu_count

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
api_key = os.getenv('AI21_API_KEY')

# Check if the API key is set
if not api_key:
    raise ValueError("API key not found. Please set the AI21_API_KEY environment variable.")

# Set up the AI21 client
client = AI21Client(api_key=api_key)

def generate_clinical_summary(row):
    """Generate a clinical summary based on the provided row of data."""
    # Build context string from the DataFrame row
    context = "\n".join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
    
    # Define the prompt
    prompt = f"""You are a helpful assistant that generates clinical notes using medical terminology.
    Utilize reasoning to analyze the following structured data and provide a detailed, narrative clinical note
    reflecting typical communication in clinical notes without suggesting solutions. Use paragraph breaks to separate different aspects of the note.

    Data: {context}

    <<Sample output of Clinical Note>>:
    The patient, with ID 10005817, was administered a single dose of IV antibiotics as per the treatment plan. The administration occurred on December 16, 2132, at 19:50, with a total volume of 500 ml delivered.

    The patient weighs 91 kg and was under the care of caregiver ID 4793. The status of the medication administration is noted as "Finished Running."
    
    Please provide a similar style of clinical note for the given data."""

    # Create the chat completion
    response = client.chat.completions.create(
        model="jamba-1.5-large",
        messages=[ChatMessage(role="user", content=prompt)],
        n=1,
        max_tokens=1024,
        temperature=0.4,
        top_p=1,
        response_format=ResponseFormat(type="text"),
    )

    # Extract and return the generated summary
    return response.choices[0].message.content

# Load the CSV file into a DataFrame
df = pd.read_csv('/inputevents.csv')

# Define the number of parallel processes
num_processes = 100

# Split the DataFrame into chunks
df_chunks = np.array_split(df, num_processes)

# Function to process a chunk of the DataFrame
def process_chunk(chunk):
    chunk['clinical_summary'] = chunk.apply(generate_clinical_summary, axis=1)
    return chunk

# Use multiprocessing to apply the function to all rows of the DataFrame
with Pool(num_processes) as pool:
    results = pool.map(process_chunk, df_chunks)

# Combine the results
df_processed = pd.concat(results)

# Save the updated DataFrame to a new CSV file
output_file = '/inputevents_with_summary.csv'
df_processed.to_csv(output_file, index=False)

print(f"Clinical summaries generated and saved to {output_file}")

# Display the first few summaries
for index, row in df_processed.head().iterrows():
    print(f"\nSummary for row {index + 1}:")
    print(row['clinical_summary'])
    print("-" * 50)
