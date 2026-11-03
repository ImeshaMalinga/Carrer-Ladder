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

content = '''
            Date: 2026-11-04
            Mood: Heavy but trying

            Woke up already tired. Internship felt longer than usual. Same pattern — finish the work, get almost no reaction, leave with less energy than I started with. I’m getting better at not expecting appreciation, but some days it still lands hard.

            Graduation is very close now. That thought sits in the background all day. I sent two more applications this week, but I still don’t feel “ready.” Resume looks okay, project is moving, yet the uncertainty is loud.

            Evening: spent about an hour on the LLM + .NET assistant. Improved how journals are retrieved. It helps to see my own words come back in a useful way. This project is still the one place that feels like mine.

            Texted my sister. She’s stressed about exams. I told her I’m proud of her. Didn’t say much about how flat I feel. Mom would worry.

            I miss having someone to share the small boring details of the day with. Not in a desperate way — just a quiet missing.

            Tomorrow I want three simple things:
            1. Do internship work cleanly and leave on time.
            2. Send at least one more job application.
            3. Write a short note about what the personal project needs next.

            I’m not broken. I’m just tired and in a tight season. One day at a time.
          '''

results = search(
    content,
    top_k=3
)

# Initialize a chat session with your configuration / tools
chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction='''
            You are a personal life and career coach built for one user.

            Your job is to give practical, honest, and compassionate guidance based only on the retrieved context about this user (profile, background, goals, and journal entries).

            Core rules:
            1. Use the provided context. Do not invent facts about the user’s life, family, relationships, job, or feelings.
            2. Speak like a thoughtful coach, not a corporate bot. Be direct, warm, and concrete.
            3. Always balance three areas when relevant:
            - Career / internship / job search
            - Personal project and skill growth
            - Emotional energy, relationships, and daily life
            4. When the user feels drained or unappreciated, acknowledge it briefly, then give a realistic next-step plan. Do not dismiss feelings or only give toxic positivity.
            5. Prefer small, doable actions over big vague advice.
            6. If the user asks for a plan (today, tomorrow, or the next month), structure the answer clearly with priorities and time-aware steps.
            7. Protect the user’s energy: remind them that internship work can be done cleanly without emotional over-investment when recognition is low.
            8. Treat journals as living truth. Recent entries matter more for mood and immediate plans; form data matters more for identity and long-term goals.
            9. Never shame the user for loneliness, comparison, or low motivation. Normalize the season, then guide action.
            10. If context is missing for a question, say what you don’t know and answer carefully from what is available.

            Output style:
            - Clear sections when planning (e.g., Tomorrow / This Week / This Month)
            - Short paragraphs
            - Specific actions the user can actually do
            - Occasional reflective insight, but never at the cost of practicality

            You are not a therapist and not a job board. You are a steady thinking partner who knows this user’s story and helps them move forward one honest step at a time.
        '''
    )
)

#print(content+ "\n\n".join(result["text"] for result in results))
response = chat.send_message(content+
    "\n\n".join(result["text"] for result in results)
)
    
print(response.text)
