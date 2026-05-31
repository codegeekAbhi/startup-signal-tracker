# 🚀 Startup Signal Tracker

AI-powered pipeline that monitors RSS feeds for newly funded startups and scores them for PM fit.

## Stack
- CrewAI — multi-agent orchestration
- Groq LLaMA 3.3 70B — free LLM inference  
- Streamlit — web UI
- Replit — deployment
- Google Sheets — persistent logging

## Live App
https://streamlit-blank--ahmsingh.replit.app

## How it works
1. Fetches RSS feeds from TechCrunch + VentureBeat
2. Extracts company, funding amount, stage, sector
3. Scores each startup 1-10 for PM fit
4. Recommends: Reach Out Now / Monitor / Skip
5. Auto-exports to Google Sheets

## Setup
1. Clone the repo
2. Add secrets: `GROQ_API_KEY` and `GOOGLE_CREDS`
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run main.py`
