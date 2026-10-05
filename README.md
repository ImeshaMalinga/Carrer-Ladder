```markdown
# LifeEcho – Adaptive Life Journal & Planner

> A personal journal that turns daily reality into coherent next-day, monthly, and yearly plans — while helping you move toward a healthier, wealthier, and more meaningful life.

---

## Vision

Most productivity tools either track the past or force rigid plans onto the future.  
LifeEcho does both intelligently:

- You write what actually happened (and what didn’t).
- The system understands your goals, patterns, and constraints.
- It generates adaptive plans that stay aligned with the life you say you want.

It is **not** a prediction engine.  
It is a high-quality, personal reflection + planning system grounded in your own data.

---

## Core Principles

- **User agency first** — Every plan can be accepted, edited, or rejected.
- **Honesty over hype** — The system never claims it can predict your future.
- **Memory with intention** — Old or irrelevant details (e.g. previous career) can be down-weighted or archived.
- **Privacy by design** — Personal data stays under user control.
- **Progressive enhancement** — Start simple (MVP), then add sophistication.

---

## Features

### MVP (Current Focus)
- Set one **Ultimate Goal** + several **Active Goals**
- Daily journal entries (“What I did” + “What I couldn’t do”)
- Automatic extraction of actions, blockers, and goal signals
- Generation of:
  - Focused schedule for tomorrow
  - Lightweight rolling 30-day plan
- Basic vector memory of journal entries
- Ability to mark older information as less relevant

### Planned (Post-MVP)
- Layered memory system (Active / Goal-Relevant / Archival)
- Automatic relevance re-weighting when goals or life context change
- Dynamic monthly & yearly trajectory views
- Weekly review ritual
- Optional light external context (career/health trends)
- Future-self conversation mode
- Local-first / fully private deployment option

---

## How It Works (High Level)

```
User writes daily journal
        ↓
LLM extracts structured signals
        ↓
Entries are vectorized + tagged (goals, domains, time, relevance)
        ↓
Planner retrieves relevant memory + current goals
        ↓
Generates tomorrow’s schedule + 30-day outline
        ↓
User reviews / edits → feedback improves future plans
```

---

## Project Status

**Phase:** MVP Design & Scaffolding  
We are currently defining the data model, extraction pipeline, and basic planner.  
No production code has been written yet.

---

## Suggested Tech Stack (MVP)

| Layer              | Recommendation                          | Notes |
|--------------------|-----------------------------------------|-------|
| Backend            | Python + FastAPI                        | Fast to iterate |
| LLM                | GPT-4o / Claude 3.5 / strong open model | Extraction + planning |
| Embeddings         | text-embedding-3-small or local         | Journal vectorization |
| Database           | PostgreSQL + pgvector                   | Or SQLite for pure local |
| Frontend           | Next.js or Streamlit/Gradio             | Streamlit for fastest MVP |
| Auth (optional)    | Simple email / local only               | Privacy first |

---

## Roadmap

### Phase 1 – MVP
1. Data model (User, Goals, JournalEntry, Plan)
2. Journal input + storage
3. LLM extraction pipeline
4. Basic goal system
5. Daily + monthly planner
6. Simple retrieval + plan generation
7. Minimal UI + feedback loop

### Phase 2 – Memory & Intelligence
- Relevance tagging and time decay
- Manual + automatic archiving of outdated context
- Better progress tracking against goals
- Weekly review flow

### Phase 3 – Depth
- Full layered memory architecture
- Yearly trajectory view
- Optional domain knowledge injection
- Future-self mode
- Mobile / voice support

---

## Privacy & Ethics

- No personal data is used to train external models.
- Users can delete any entry or entire history at any time.
- The system is designed to be runnable fully locally.
- Plans are suggestions, never prescriptions.
- External knowledge (when added) is treated as soft context only.

---

## Getting Started (Once Code Exists)

```bash
# Clone the repository
git clone <repo-url>
cd lifeecho

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env

# Run the application
uvicorn app.main:app --reload
```

---

## Contributing

This project is currently in early private development.  
Contribution guidelines will be added after the MVP is stable.

---

## License

TBD (likely MIT or Apache 2.0 once public)

---

## Acknowledgments

Born from the idea of turning personal journals into living, adaptive guidance systems rather than static records.
```