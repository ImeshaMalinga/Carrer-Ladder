import os
from os import path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import glob
from data_vect import build_vector_store, embed_texts, load_documents, search

load_dotenv()
api_key = os.getenv("Gemini_API_Key")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=5,                    # total tries (1 original + 4 retries)
            initial_delay=1.0,             # start with 1 second
            max_delay=60.0,                # never wait more than 60s
            exp_base=2.0,                  # double each time
            jitter=1.0,                    # add randomness
            http_status_codes=[408, 429, 500, 502, 503, 504],  # UNAVAILABLE = 503
        )
    )
)

content = '''My internship is over and company have not decided to hire me or not, so until they decide they told me to apply for
            other companies as well, they will give their decision after 2-3 weeks, and i am doing assignments, projects as my part
            time income source and i have got to do a research on cyber security side due on december 01 , i have not done anything on 
            that yet.i feel stressed and i couldn't do any work because of that
            '''

results = search(
    content,
    top_k=3
)

# Initialize a chat session with your configuration / tools
chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a carrer builder, give a paln for tommorow and for next month",
    )
)

#print(content+ "\n\n".join(result["text"] for result in results))
response = chat.send_message(content+
    "\n\n".join(result["text"] for result in results)
)
    
print(response.text)
