# Shibu AI — Rule-Based Intelligent Chatbot

A complete, runnable university-level project: a **rule-based intelligent agent**
named **Shibu AI** (cute anime avatar in the top-left corner, `static/shibu.png`)
in Python + Flask, with SymPy maths, DuckDuckGo web search, ReportLab PDF
export, session memory, and explainability details. The UI is clean and
Muse-style: a black background, a centred chat column, and just one text
bar at the bottom. Explainability (intent, rule, confidence, tool, time)
and the agent-environment summary live behind a small "Details" toggle
in the header, so the main view stays uncluttered.

> This is deliberately **not** just an LLM chatbot. The rule engine decides
> first. Hugging Face is optional and never replaces the rules.

## 1. Objectives
- Build a demonstrable intelligent agent (perception → state → decision → action).
- Show explainable decisions: intent, matched rule, confidence, tool, time.
- Add useful tools without paid APIs: maths, web search, PDF.
- Keep the project runnable with one command: `python app.py`.

## 2. Features
- 80 rules: greetings, thanks, Python, AI, ML, DL, NLP, CS, Flask, APIs, project/viva help, plus Java, JavaScript, C/C++, HTML/CSS, Git, OOP, operating systems, networks, cybersecurity, cloud, blockchain, robotics, IoT, statistics, calculus, SQL, interview/resume/study tips, and more, with synonyms ("hi"/"hey" → greeting).
- Confidence: exact = 0.98, regex ≈ 0.90, keywords up to 0.85, synonyms up to 0.65; labelled HIGH / MEDIUM / LOW and shown in the UI.
- Session context: "What is Python?" then "What is it used for?" works.
- Maths (SymPy, never `eval()`): `25 + 50`, `25% of 800`, `sqrt(144)`, `2^10`, `solve x + 5 = 12`, `derivative of x^3`, `integral of x^2`, mean/median.
- Web search (DuckDuckGo, free): title, URL, snippet, source — always labelled as web-derived.
- PDF (ReportLab): topic PDF, conversation PDF, any text; title, timestamp, page numbers, footer; files in `generated/` with a download button.
- Conversational intelligence: follow-ups, multi-step instructions ("What is Python? then calculate 25% of 800"), clarifying questions for vague input instead of guessing, and saved preferences ("set tone to simple", "expert level", "explain simply"). Low-confidence answers are labelled as uncertain.
- Reasoning: step-by-step maths ("Solve step by step 25 + 50"), code generation templates ("Write Python code for factorial"), code debugging with AST syntax checks ("Debug this code: ..."), and comparisons with trade-offs ("Compare Python and Java").
- Document intelligence: upload PDF/DOCX/XLSX with the paperclip button, then "Summarize my document", ask questions about it, "Make notes from my document" (study guide PDF), and spreadsheet trend analysis (count, sum, mean, min, max, increasing/decreasing).
- File generation: "Create a PowerPoint about Python", "Create a Word document about AI", "Create an Excel workbook about Sales" (table + SUM formulas + bar chart), plus PDF. Or use `/api/generate-file` with `type` = pptx/docx/xlsx/pdf, `title` and `content`. The user specifies format, length and style in plain language.
- Safe Python runner: "Run Python code: print(2 + 3)" runs your own code in a separate process with a 3-second timeout. Only math/random/statistics/datetime imports are allowed; os/sys/subprocess/file access are blocked. Only run code you wrote and trust.
- UI: clean centred chat, one text bar, typing indicator, auto-scroll, Enter to send / Shift+Enter for a new line, copy link, download link for PDFs, clear chat, responsive layout. Voice input via the mic button in the text bar (Chrome/Edge), and chat history saved in your browser so reopening the page restores it. A **History** section in the header lists your past questions — click one to ask Shibu AI again. Web search answers are phrased as a summary ("Here's what I found on the web about...") followed by numbered sources, never as a raw list. Maths, search and PDF still work by typing, e.g. "Calculate 25% of 800", "Convert 5 km to miles", "Search latest AI news", "Create a PDF about Python".

## 3. Architecture — Intelligent Agent Model
```
User
 ↓  Perception: receive the message
State: current input + conversation history
 ↓  Decision: rule engine (regex + keywords + synonyms + priority + confidence)
Action: rule answer  |  SymPy maths  |  DuckDuckGo search  |  ReportLab PDF
 ↓  Response (with explainability)
Browser / user interaction (Environment)
```

## 4. Environment Classification
- **Observability: Partially Observable** — only user messages + limited history are visible.
- **Determinism: Deterministic** — identical input + state normally produce the same rule decision.
- **Structure: Sequential** — previous conversation influences future responses.
- **Dynamics: Static** — the rule environment does not change during normal interaction.

This is also shown behind the "Details" toggle in the UI header.

## 5. Technology Stack
Python 3.11+, Flask, Flask-CORS, python-dotenv, SymPy, ReportLab, requests, (optional) ddgs + Hugging Face. Frontend: HTML5, CSS3, JavaScript (custom black theme, no Bootstrap dependency needed).

## 6. Folder Structure
```
rule_based_chatbot/
  app.py  requirements.txt  README.md  .env.example  .gitignore
  chatbot/  rule_engine.py patterns.py intents.py router.py context.py confidence.py response_manager.py
  services/ math_service.py huggingface_service.py pdf_service.py web_search_service.py
  models/ schemas.py
  utils/ logger.py security.py text_processing.py
  templates/index.html
  static/css/style.css  static/js/app.js
  generated/.gitkeep
  tests/ ...
  PROJECT_DOCUMENTATION.md  VIVA_QUESTIONS.md  PRESENTATION_OUTLINE.md
```

## 7. Installation (Windows, macOS, Linux)
```bash
cd rule_based_chatbot
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
pip install -r requirements.txt
```

## 8. Environment Variables
```bash
cp .env.example .env
# Optional only:
# HF_TOKEN=hf_...        (leave empty = Hugging Face disabled, app still works)
# HF_MODEL=google/flan-t5-small
# SECRET_KEY=any-random-string
# PORT=5000
```
Never commit `.env`. Never hard-code tokens.

## 9. Running the Application
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

## 10. API Documentation
| Method | Route | Body | Returns |
|---|---|---|---|
| GET | `/` | — | Chat UI |
| POST | `/api/chat` | `{"message": "..."}` | response, intent, rule_id, confidence, confidence_label, tool, action, processing_ms, plus pdf_url / search_results when relevant |
| POST | `/api/calculate` | `{"expression": "25% of 800"}` | calculation, answer |
| POST | `/api/search` | `{"query": "latest AI news"}` | results [{title, url, snippet, source}] |
| POST | `/api/generate-pdf` | `{"content": "...", "title": "..."}` (content optional = whole conversation) | pdf_file, pdf_url |
| GET | `/api/download/<filename>` | — | Generated file (PDF/PPTX/DOCX/XLSX) |
| POST | `/api/upload` | multipart `file` (PDF/DOCX/XLSX) | filename, info, summary preview; then chat about it |
| POST | `/api/generate-file` | `{"type": "pptx", "title": "...", "content": "..."}` | file, file_url |
| GET/POST | `/api/preferences` | `{"tone": "simple", "level": "beginner"}` | saved preferences |
| GET | `/api/llm/status` | — | LLM mode, model, RAM, GPU, cache (never the token) |
| POST | `/api/llm/chat` | `{"message": "..."}` | direct LLM answer (503 if no model enabled) |
| POST | `/api/llm/test` | — | one-sentence LLM smoke test |
| GET | `/api/history` | — | conversation context |
| POST | `/api/clear` | — | `{"status": "cleared"}` |
| GET | `/api/health` | — | status, rule_engine, math_service, pdf_service, search_service, huggingface |

Errors use proper codes: 400 for empty/invalid input, 500 for server errors, without exposing tracebacks.

## 11. Mathematics Module
`services/math_service.py` uses SymPy only. Percentage, derivative, integral, solve, factor and statistics are parsed with regex first, then `sympy.sympify` with a restricted local namespace. `eval()` is never used on user input.

## 12. Hugging Face Language Model (optional, rule-first)
Architecture, in the existing structure instead of a duplicate one:
- `services/model_manager.py` — `model_manager`: selection, cache check, lazy loading, API + local generation, plain-language errors. Nothing is downloaded at startup.
- `services/conversation_manager.py` — `conversation_manager`: last 6 turns, trimmed to ~6,000 characters, Hindi/Hinglish detection, system prompt + chat-template use.
- `services/response_generator.py` — `response_generator`: validates (empty = failure, repeated answer = failure), keeps uncertainty, labels every AI answer.
- `chatbot/router.py` — `intent_router`: rules, maths, search, documents and files run first. The LLM only answers UNKNOWN / low-confidence questions, and explicit `ai: ...` messages.
- `services/web_search_service.py` — `web_search` (already existed, unchanged in purpose): DuckDuckGo with sources and links.
- `services/document_service.py` — `document_processor` (already existed): PDF/DOCX/XLSX extraction, summary, Q&A, notes, spreadsheet trends.
- `services/file_generation_service.py` — `file_generator` (already existed): PPTX/DOCX/XLSX/PDF downloads.

How it is enabled (pick one; the old `services/huggingface_service.py` API path is kept for compatibility):
- **Option A — API (recommended, easiest):** copy `.env.example` to `.env`, set `HF_TOKEN` to a free token from huggingface.co/settings/tokens. The token must have the **"Make calls to Inference Providers"** permission — a plain read-only token without it is rejected with HTTP 401. No download, no extra RAM. Default API model: Qwen/Qwen2.5-3B-Instruct (set `HF_MODEL` to change it). API calls go to the current Inference Providers router, `https://router.huggingface.co/v1/chat/completions` — the old `api-inference.huggingface.co` endpoint is decommissioned and no longer connects, so do not use old tutorials that point at it. If an `ai:` message fails, Shibu now shows the real Hugging Face error (400 = model not supported by any provider you have enabled, 401 = token/permission, 404 = model not on the router, 429 = rate limit/credits, 503 = model loading) instead of saying "not enabled". For a 400, Shibu automatically retries the same model as `model:auto` and then, for Qwen models, as `model:featherless-ai` (the only live provider for Qwen2.5-3B-Instruct, verified against the Hub API), and labels the answer with the model+provider that actually answered. You can force a provider with `HF_PROVIDER=featherless-ai` in `.env`, or enable it at huggingface.co/settings/inference-providers.
- **Option B — local model:** `pip install -r requirements-llm.txt`, then set `HF_USE_LOCAL=1` in `.env`. The model is auto-selected from RAM (see below) and downloaded once, on first LLM use, into the Hugging Face cache — later startups reuse it and download nothing. Set `HF_MODEL` to override the auto-pick, and `HF_QUANTIZE=4bit` (with bitsandbytes) to try a larger model quantized.

Model selection (verified 2026-10-24: all public, not gated, no login needed to download):
| RAM | Auto-pick | Why |
|---|---|---|
| ≤ 8 GB | Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0, ~1 GB) | 3B/7B risk out-of-memory here |
| 8–16 GB | Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0, ~3 GB) | 3B only quantized |
| 16 GB+ | Qwen/Qwen2.5-3B-Instruct (Qwen licence, ~6 GB), or 7B with a GPU | Needs the memory to spare |

Alternatives evaluated: microsoft/Phi-4-mini-instruct (MIT, ~3.8B, ~8 GB, mainly English) — a good 16 GB+ alternative, not a Hindi-first choice. Qwen models support many languages including Hindi to a useful degree; they are not perfect in Hindi, so Hindi answers from the rule engine (e.g. "Namaste") and the LLM's language instruction are both kept.

Honest limits: a 0.5B/1.5B model is a study assistant, not a frontier model — it will be weaker than the API mode on hard reasoning, and it can still be wrong, which is why AI answers are labelled and uncertainty is kept. If the model fails (no token, no local install, out of memory, network down), the chatbot keeps working from its rules and says the LLM is unavailable, instead of crashing or inventing an answer.

Try it:
- `GET /api/llm/status` — shows mode, model, RAM, GPU, cached, and how to enable (never the token).
- `POST /api/llm/test` — one-sentence smoke test (in local mode this triggers the one-time download).
- In chat: `ai: explain photosynthesis simply`, or an unknown question with the LLM enabled.
- Hindi/Hinglish: `Namaste` works from the rules even with no LLM; with the LLM enabled, `ai: photosynthesis kya hai?` should answer in Hinglish.

## 13. Web Search
`services/web_search_service.py` tries the `ddgs` library, then falls back to the DuckDuckGo HTML endpoint. On failure it returns `[]` and the UI says so honestly.

## 14. PDF Generation
`services/pdf_service.py` (ReportLab Platypus) adds title, timestamp, headings, page numbers and footer, and writes only into `generated/` with sanitised filenames.

## 15. Testing
```bash
pytest -v
```
Tests cover rules, synonyms, confidence range, context, maths (2+3=5, 25% of 800=200, sqrt(144)=12), PDF generation, mocked search, and Flask routes.

## 16. Troubleshooting
- **Port 5000 in use:** set `PORT=5001` in `.env` and run again, or stop the other app.
- **ModuleNotFoundError:** make sure the virtual environment is activated and `pip install -r requirements.txt` finished.
- **Search returns nothing:** check internet; DuckDuckGo may rate-limit briefly — try again.
- **PDF won't download:** check the `generated/` folder exists and the file name in the URL matches.
- **Hugging Face false in health:** expected if you didn't set `HF_TOKEN` — the app is designed to work without it.
- **LLM out of memory:** use a smaller model (`HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct`), set `HF_QUANTIZE=4bit` (needs bitsandbytes), or switch to the API mode with `HF_TOKEN`.
- **LLM test returns 503:** no model is enabled — that is the documented fallback, not a crash. Follow the `/api/llm/status` "how_to_enable" instructions.

## 17. Future Enhancements
Database persistence, user login, voice input/output, RAG over documents, larger knowledge base, better NLP, full LLM integration as an optional layer.

## Screenshots
*(Add screenshots here after running the app — e.g. `docs/screenshot-chat.png`, `docs/screenshot-pdf.png`.)*

## 18. GitHub, Cloud, and ChatGPT API (fast response)

**ChatGPT API:** Set `OPENAI_API_KEY` in `.env` (locally) or in your cloud host's Environment Variables (never in git, never in chat). Default model is `gpt-4o-mini` — fast and cheap; change it with `OPENAI_MODEL`. When this key is set, Shibu uses OpenAI first for UNKNOWN / `ai:` questions, before Hugging Face, and labels the answer with the model. If the key is missing or invalid, Shibu falls back to Hugging Face / rules and shows the real OpenAI error (401 = bad key, 429 = rate/credits, 404 = bad model name). Get a key at platform.openai.com — it is a paid API, so check your OpenAI billing/credits; a free Hugging Face token is the no-cost alternative.

**GitHub:** The project is ready to push — `.gitignore` excludes `.env`, `venv/`, `generated/` and `uploads/`, so no secrets or outputs are committed. Create a repo on GitHub (e.g. `shibu-ai`), then push the project folder. Never commit `.env`.

**Cloud (Render, free plan):** The project includes `Procfile` (`web: python app.py`), `render.yaml`, and `runtime.txt` (Python 3.12). On Render: New > Web Service > connect your GitHub repo, it will use `render.yaml` (build: `pip install -r requirements.txt`, start: `python app.py`). Then in Render's Environment tab, set `SECRET_KEY`, and either `OPENAI_API_KEY` or `HF_TOKEN` — do not put keys in the repo. The app listens on `0.0.0.0:$PORT`, which is what a cloud host requires (locally it still opens at http://127.0.0.1:5000). Note: Render's free plan sleeps after inactivity, so the first visit after a pause can take ~30-60 seconds to wake up. Uploaded documents and generated files are stored on the server's disk, so they do not survive a redeploy/restart on a free plan — download anything important first.

## 19. New topics, conversation, and text-processed web answers (2026-10-27)

**Topics now covered (153 rules total):** Science (physics, Newton's laws, gravity, energy, light, sound, chemistry, atom, periodic table, acids/bases, biology, cell, photosynthesis, human body, ecosystem), Social Studies (history, India's independence, geography, continents/oceans, climate, civics/democracy, Constitution of India, economics, demand/supply), GK (capital of India, largest planet, tallest mountain, national symbols, first President/PM, longest river, fastest animals), Maths topics (algebra, geometry, trigonometry, fractions, percentages, ratio, average, prime numbers, number system), Reasoning (logical reasoning, deductive/inductive, series, coding-decoding, blood relations, direction sense), English (parts of speech, noun, verb, tenses, active/passive, articles/prepositions, synonym/antonym), and Hindi (Hindi language, sandhi, samas, paryayvachi/vilom, sangya/kriya, varnamala).

**Conversation:** Shibu can now chat, not just answer FAQs — hobbies, favourites, "are you human?", friend, bored, motivation, age, where it lives, stories, "and you?", "interesting", "talk to me". Try: "Synonym of happy", "Antonym of big", "Next number in series 2, 4, 6, 8".

**Text processing for web answers (the main change):** Raw DuckDuckGo snippets are now cleaned (HTML/whitespace removed), split into sentences, scored by query-keyword coverage, deduplicated, and presented as a short summary + key points + sources (see `services/web_answer.py` and `utils/text_processing.py`). Most importantly, the old dead-end — "I don't have a specific rule for that term yet" — is now a web fallback: if a "what is X" question has no rule and no LLM, Shibu searches the web, processes the results, and returns a labelled web answer with sources. If the web also returns nothing, it falls back to the honest rule message. Web answers are never presented as rule-database answers.

## 20. AI Environment Analyzer + Conversation Context Memory (2026-10-27)

**Environment analyzer (`services/environment_analyzer.py`):** Ask `Classify a self-driving car environment.` and Shibu returns a table — Observability, Predictability, Task Structure, Environment Dynamics — each with one scenario-specific reason. Classic profiles cover self-driving car, chess, tic-tac-toe, poker, robot vacuum, warehouse robot, crossword, spam/image classifier, and Shibu AI itself (Partially Observable · Deterministic · Sequential · Static, as in §4). Other scenarios are reasoned from your own description (hidden sensors, randomness, moving agents, time-decaying score); a property with no evidence is shown as **Undetermined** with what is missing, not guessed. The table is validated before rendering, and if the analyzer fails the normal answer still shows. It never fires on unrelated questions: a plain topic question gets no table, and analyzing a scenario is kept separate from analyzing Shibu AI itself (ask `Classify Shibu AI's environment.` for the latter). In the UI the table renders as a separate section below the main answer (see `static/js/app.js`, `.env-table` in `static/css/style.css`); API clients get it in the `environment_analysis` field of `/api/chat`.

**Conversation context memory (`services/context_memory.py` + `chatbot/context.py`):** History is still per-session (Flask session id, `ContextStore`, cleared by `/api/clear` and the Clear button — never shared between chats). On top of it, follow-ups now work: `Explain it more simply.`, `Make it shorter.`, `Give me another example.`, `What about the second one?`, `Continue from where you stopped.`, and `it / that topic` references all resolve against the previous real answer (follow-up wrappers never become the new "previous answer", so chains don't degrade). A bare reference with no previous answer, or an ambiguous "second one" with no second example, gets a short clarification question instead of a guess. A genuinely new topic is answered on its own — old context is kept in history but never injected. A compact rolling summary (last 3 topics, ≤500 chars, in `/api/history` as `summary`) keeps long conversations on track without growing the context window. No passwords or keys are stored in memory.

**Pipeline:** message → recent history + summary → follow-up resolution → normal answer (rules/LLM/web, unchanged) → environment analysis appended only when the message actually describes an environment → save message, answer and updated summary to that session.
