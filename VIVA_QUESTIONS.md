# Viva Questions — Rule-Based Intelligent Chatbot (30+ with answers)

1. **What is a rule-based system?** A system that decides by matching input against explicit rules (patterns, keywords, conditions) rather than learning from data alone. Advantage: predictable and explainable.
2. **What is an intelligent agent?** An entity that perceives its environment, maintains state, decides, and acts to achieve a goal. Here: message → state → rule decision → answer/tool.
3. **Explain perception, state, decision, action in this project.** Perception = receiving the message; State = message + history + last intent/topic; Decision = rule engine + confidence; Action = answer or maths/search/PDF tool call.
4. **What does partially observable mean here?** The bot sees only messages and limited history, not the user's full intent, mood, or environment.
5. **Why is this environment deterministic?** Same input + same state normally yields the same rule and the same (first) response — unlike a stochastic LLM.
6. **What is a sequential environment?** Earlier actions/messages affect later ones — e.g. a follow-up "What is it used for?" depends on the previous topic.
7. **What is a static environment?** The rules/environment do not change during a normal chat (no new rules are learned mid-conversation).
8. **What is regex and why use it?** Regular expressions match text patterns, e.g. `\bhello\b`. They give high-confidence, precise matches before looser keyword matching.
9. **How does NLP appear in a rule-based bot?** Lightweight NLP: normalisation (lowercase, whitespace, punctuation), synonym replacement, keyword extraction.
10. **How is confidence calculated?** Exact 0.98, regex ~0.90, keyword coverage up to 0.85, synonym up to 0.65, clamped 0–1, labelled HIGH/MEDIUM/LOW.
11. **Why show confidence in the UI?** Explainability — the user/examiner can see how sure the bot is and which rule fired.
12. **What is state/action in agent terms?** State is what the agent remembers; action is what it does next (answer, calculate, search, generate PDF).
13. **Why not use eval() for maths?** `eval()` would execute arbitrary Python from user input — a security hole. SymPy parses maths safely.
14. **What does SymPy do here?** Symbolic maths: arithmetic, percentages, sqrt, powers, solving, factoring, derivatives, integrals, statistics.
15. **What is Flask?** A lightweight Python web framework for routes, templates and JSON APIs.
16. **What is a REST API?** HTTP-based interface using methods like GET/POST and JSON. This app exposes /api/chat, /api/calculate, /api/search, /api/generate-pdf, /api/health, etc.
17. **How does session context work?** Flask session id → in-memory ContextStore → ConversationContext storing history, last intent/rule/confidence/topic. No database needed.
18. **How does "What is it used for?" work after "What is Python?"** The router detects a follow-up pattern, looks up last_topic="python", and returns the stored "used for" answer for that topic.
19. **What is Hugging Face's role?** Optional enhancement for UNKNOWN answers only, configured by HF_TOKEN/HF_MODEL. It never overrides the rule engine.
20. **How is PDF generation done?** ReportLab Platypus with title, timestamp, page numbers, footer, writing only to generated/ with sanitised filenames.
21. **How does web search work without a paid API?** DuckDuckGo (ddgs library, HTML fallback), returning title, URL, snippet, source.
22. **How do you distinguish web results from rule answers?** Results are labelled as web-derived in text and in the UI, with sources shown.
23. **What happens on an unknown query?** Graceful fallback: the UNKNOWN rule suggests help, rephrasing, or web search, instead of crashing or hallucinating.
24. **What happens on invalid input?** Validation returns HTTP 400 with a friendly message; server errors return 500 without tracebacks; details are logged server-side.
25. **How are secrets handled?** Via .env (python-dotenv), never hard-coded, .env is git-ignored, and logs never include tokens.
26. **How are filenames made safe?** Sanitisation strips path characters and safe_join rejects traversal outside generated/.
27. **What are the limitations?** Finite rules, no persistent memory, search needs internet, no voice/login, no open-ended reasoning without HF.
28. **What would you add next?** Database, auth, voice, RAG, larger knowledge base, better NLP.
29. **Why is priority needed in rules?** To break ties: e.g. "Python used for" (priority 45) should beat generic "Python" (40).
30. **How do synonyms help?** "hi", "hey", "hiya" all normalise to the greeting intent, so one rule covers many phrasings.
31. **What is the difference between rule-based and ML chatbots?** Rules are hand-written, deterministic and explainable; ML bots learn from data, handle varied phrasing better, but are less predictable and harder to audit.
32. **How would you test this?** pytest: rules, synonyms, confidence range, context, maths, PDF file creation, mocked search, Flask routes.
33. **What is CORS and why is it enabled?** Cross-Origin Resource Sharing allows the browser frontend to call the Flask API; Flask-CORS handles it safely for this local project.
