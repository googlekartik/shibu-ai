# Project Documentation — Rule-Based Intelligent Chatbot

## Abstract
This project builds a complete, runnable rule-based intelligent chatbot in
Python and Flask. A rule engine using regex, keywords, synonyms, priority and
confidence decides every answer, and calls tools for mathematics (SymPy),
current information (DuckDuckGo search) and PDF export (ReportLab). Session
memory allows follow-up questions. An optional Hugging Face model can enhance
unknown answers but never replaces the rule engine. The web UI is black-themed
and modern, with an explainability panel showing intent, rule, confidence,
tool and processing time for every response.

## Introduction
Rule-based systems were the first practical chatbots: they are predictable,
testable and explainable. Intelligent agents perceive an environment, keep
state, decide, and act. This project combines both ideas — a chatbot as an
agent whose decisions can be inspected, which is exactly what a university
examiner wants to see.

## Objectives
1. Implement a real rule engine, not just if/else greetings.
2. Add confidence and explainability.
3. Add maths, search and PDF tools safely (no `eval()`, no paid APIs).
4. Support conversation context and follow-ups.
5. Deliver a polished, runnable web application with tests and documentation.

## Problem Statement
LLM-only chatbots are powerful but opaque, costly and non-deterministic.
For teaching, customer FAQs and controlled domains, a rule-based agent gives
deterministic, auditable answers. The problem is making such a bot substantial
enough — 50+ rules, real confidence, tools, memory and a modern UI — rather
than a 10-line demo.

## System Architecture
- **Frontend:** `templates/index.html`, `static/css/style.css`, `static/js/app.js` — black theme, chat + explainability panel.
- **Backend:** `app.py` — Flask routes for chat, calculate, search, PDF, history, clear, health, download.
- **Chatbot core:** `chatbot/` — patterns (knowledge base), rule_engine, confidence, context, router (pipeline), intents, response_manager.
- **Services:** `services/` — math (SymPy), search (DuckDuckGo), PDF (ReportLab), Hugging Face (optional).
- **Utils:** text normalisation, logging, security/validation.

## Methodology
Input → Validation → Text Normalisation → Context Retrieval → Intent
Detection → Rule Matching → Confidence Calculation → Decision → Tool
Selection (rule / maths / search / PDF) → Response Formatting → Context
Update → Web UI. The router (`chatbot/router.py`) implements this order:
maths and PDF and search are checked before generic rules, so "Calculate
sqrt(144)" is calculated, not answered with a generic maths description.

## Intelligent Agent Characteristics
- **Partially Observable:** only messages + limited history are observed.
- **Deterministic:** same input + state → same rule decision (first response per rule is used deliberately for reproducibility).
- **Sequential:** history influences follow-ups ("it" = previous topic).
- **Static:** rules do not change during a chat.

## Rule Matching Algorithm
1. Normalise: lowercase, collapse whitespace, remove punctuation, apply synonyms.
2. For each rule (except "unknown"): count regex matches, keyword hits, synonym hits.
3. Compute confidence (see below), add priority as a tie-breaker.
4. Pick the highest-scoring rule; if none matches, return the UNKNOWN rule with a graceful fallback.

## Confidence Algorithm
Exact phrase → 0.98. Regex match → 0.88–0.95. Keyword coverage → 0.35–0.85.
Synonym match → 0.40–0.65. Clamped to 0.0–1.0 and labelled HIGH (≥0.75),
MEDIUM (≥0.45) or LOW (otherwise). Maths answers use 0.99 by design.

## Mathematics Module
SymPy parses expressions after regex handling of percentages, derivatives,
integrals, equations, factors and statistics. No `eval()`. Invalid input
returns a friendly error, not a traceback.

## Hugging Face Integration
Optional, configured by `HF_TOKEN`/`HF_MODEL`. Used only for UNKNOWN intents
as an "[AI-assisted suggestion]". Without a token the system is fully
functional and `/api/health` reports `huggingface: false`.

## Web Search
DuckDuckGo via the `ddgs` library, with an HTML-endpoint fallback. Every
result shows title, URL, snippet and source, and the UI labels it as
web-derived, never as rule knowledge.

## PDF Generation
ReportLab Platypus: title, timestamp, clean typography, page numbers and
footer. Output is restricted to `generated/` with sanitised filenames.

## Results — Example Interactions
- "Hey" → greeting, HIGH confidence.
- "What is Python?" → Python definition; then "What is it used for?" → context-aware Python uses.
- "What is 25% of 800?" → 200.
- "What is the derivative of x^3?" → 3*x**2 (shown as 3x² in the UI text).
- "Search the latest AI developments." → sourced web results.
- "Create a PDF about Python." → downloadable PDF.
- "Some completely unsupported query." → graceful fallback with suggestions.

## Limitations
Rule coverage is finite; phrasing far from any rule falls back. Search
depends on internet/DuckDuckGo availability. Memory is per-session and lost
on clear/restart (no database by design). No voice, no login, no persistent
history — all listed as future scope.

## Future Scope
Database, user authentication, voice input/output, better NLP, RAG,
larger knowledge base, LLM integration as an optional layer.
