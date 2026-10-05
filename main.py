import os
from os import path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import glob

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

content = '''I am a software engineering ug , out in one month ,
            finding a carrer(job), today i have started new project based on LLM and .NET,
            but i could focusly do my internship work thorough out my day time because enough appretiation
            for my work at my internship work place.
            '''
# Initialize a chat session with your configuration / tools
chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a carrer builder, give a paln for tommorow and for next month",
        # tools=[your_tool_function]  <-- If using tools
    )
)


response = chat.send_message(content)
    
print(response.text)
