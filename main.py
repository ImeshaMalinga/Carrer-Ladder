import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
from data_vect import (
    add_embeddings_to_store,
    chuck_text,
    embed_texts,
    load_documents,
    search,
)

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
            Date: 2026-11-05  
            Mood: Quiet and a little heavier than yesterday

            Woke up later than I meant to. The tiredness didn’t fully leave overnight. Internship was slower today — fewer tasks, more waiting. I finished what was given and left on time, which felt like a small win even if no one noticed.

            Sent one more application in the afternoon. Still no replies from the earlier ones. The silence is starting to feel familiar. Graduation is only weeks away now, and the gap between “almost done” and “what comes next” feels wider than it should.

            Worked on the LLM + .NET project for a bit longer than planned. Fixed a small retrieval issue and cleaned up how journal entries are ranked. Seeing my own words come back still feels grounding. It’s the one place where progress is visible and belongs only to me.

            Spoke with my sister briefly. She’s still buried in exams. I mostly listened. Didn’t mention how flat the day felt. Some things are easier to carry quietly.

            The quiet missing is still there — not sharp, just present. A low background hum.

            Tomorrow I want three simple things:  
            1. Finish internship tasks without dragging them past the end of the day.  
            2. Spend focused time on the personal project (even 45 minutes counts).  
            3. Write down one concrete next step for the job search instead of just “apply more.”

            I’m still here. Still moving, even if the pace is slower than I want. One day at a time.
          '''

journal_path = Path("data/journals/2026-11-05.txt")
journal_path.parent.mkdir(parents=True, exist_ok=True)
journal_path.write_text(content.strip() + "\n", encoding="utf-8")
print(f"Created journal: {journal_path}")

documents = load_documents(str(journal_path))
chunks = chuck_text(documents)

chunk_texts = [chunk["text"] for chunk in chunks]
chunk_embeddings = embed_texts(
    chunk_texts,
    task_type="RETRIEVAL_DOCUMENT",
)

retrieved_context = []
print("Retrieved chunks (k=1 for each journal chunk):")
for chunk in chunks:
    results = search(chunk["text"], top_k=1)
    print(f"\nQuery: {chunk['text']}")
    if results:
        match = results[0]
        print(f"  Match: {match['text']}")
        print(f"  Category: {match['metadata'].get('category')}")
        print(f"  Distance: {match['distance']:.4f}")
        retrieved_context.append(match["text"])


add_embeddings_to_store(chunks, chunk_embeddings)

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

#print(content + "\n\n".join(retrieved_context))
response = chat.send_message(
    content + "\n\n" + "\n\n".join(retrieved_context)
)

print("Response from AI Coach:")

print(response.text)
