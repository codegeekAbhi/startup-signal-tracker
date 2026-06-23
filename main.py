# ============================================
# Startup Signal Tracker - CrewAI Pipeline
# Phase 1: RSS Fetch - Extract - Score
# ============================================

import os
import json
import re
import threading
import email.utils
import litellm
import feedparser
import pandas as pd
import gspread
import streamlit as st

from datetime import datetime, timezone, timedelta
from google.oauth2.service_account import Credentials
from crewai import Agent, Task, Crew, Process, LLM


# ============================================
# CONFIG
# ============================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GOOGLE_CREDS = os.environ.get("GOOGLE_CREDS", "")
SHEET_NAME = "Startup Signal Tracker"

RSS_FEEDS = {
    "TechCrunch Venture": "https://techcrunch.com/category/venture/feed/",
    "TechCrunch Startups": "https://techcrunch.com/category/startups/feed/",
    "VentureBeat": "https://venturebeat.com/feed/",
}

FUNDING_KEYWORDS = [
    "raises", "raised", "funding", "seed", "series a", "series b",
    "backed", "million", "launches", "announces"
]

EXCLUDE_PHRASES = [
    "applications close", "third fund", "closed a fund",
    "retail venture ipo", "doubles valuation", "disrupt 2026",
    "50% off", "get ready for", "hottest place", "nvidia has",
    "google's new", "google says", "google just", "google unveils",
    "openai co-founder", "anthropic warns", "apple unveils",
    "stage at techcrunch",
]


# ============================================
# LLM SETUP
# ============================================

os.environ["LITELLM_CACHE"] = "False"
os.environ["LITELLM_ENABLE_CACHING"] = "False"
os.environ["GROQ_CACHE"] = "False"
litellm.cache = None
litellm.caching = False

llm = LLM(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def parse_date(date_str):
    try:
        return email.utils.parsedate_to_datetime(date_str).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def has_funding_signal(title):
    t = title.lower()
    return any(kw in t for kw in FUNDING_KEYWORDS)


def not_noise(title):
    t = title.lower()
    return not any(phrase in t for phrase in EXCLUDE_PHRASES)


def fetch_and_filter_rss():
    raw_entries = []
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            raw_entries.append({
                "source": source,
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
            })

    df = pd.DataFrame(raw_entries)
    df = df.drop_duplicates(subset="title")
    df["parsed_date"] = df["published"].apply(parse_date)

    now = datetime.now(timezone.utc)
    first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    first_of_last_month = (first_of_this_month - timedelta(days=1)).replace(day=1)

    df = df[df["parsed_date"].notna() & (df["parsed_date"] >= first_of_last_month)]
    df = df[df["title"].apply(has_funding_signal)]
    df = df[df["title"].apply(not_noise)]
    df = df.sort_values("parsed_date", ascending=False).reset_index(drop=True)
    df["date"] = df["parsed_date"].dt.strftime("%Y-%m-%d")

    return df, first_of_last_month, now


def build_agents():
    scout = Agent(
        role="Startup Signal Scout",
        goal="Fetch RSS feeds, filter for recently funded startups, return clean deduplicated entries from the current and last month.",
        backstory="Expert at monitoring startup news, spotting genuine funding announcements and filtering noise.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    researcher = Agent(
        role="Startup Data Researcher",
        goal=(
            "Extract precise structured data from each startup funding entry. "
            "You must extract the EXACT company name as written in the headline, not a description. "
            "Also extract investor/VC names and any founder, CEO, or CTO names mentioned."
        ),
        backstory=(
            "You are a precise data extractor who reads startup funding headlines carefully. "
            "You never replace a company name with a generic description. "
            "If the headline says 'Acme raises $10M', the company is Acme, not 'a startup'. "
            "You also identify which VC firms or investors backed the round, and any key people mentioned."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    analyst = Agent(
        role="PM Fit Analyst",
        goal="Score each startup 1-10 for PM fit and recommend: reach out now, monitor, or skip.",
        backstory=(
            "Evaluates startups for a senior PM candidate: 7+ years at Amazon/Deloitte/TCS, "
            "B2B SaaS and AI experience, MBA UC Davis 2026, targeting Series A/B. "
            "Low fit for Hardware and consumer social."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    return scout, researcher, analyst


def build_tasks(scout, researcher, analyst):
    scout_task = Task(
        description=(
            "Fetch RSS feeds from:\n"
            "- https://techcrunch.com/feed/\n"
            "- https://techcrunch.com/category/startups/feed/\n"
            "- https://venturebeat.com/feed/\n\n"
            "Filter by funding keywords, current month and last month, deduplicate, exclude noise. "
            "Return each entry with: title (exact headline text), summary, published date, and link. "
            "Sort entries by published date, newest first. "
            "Do NOT paraphrase or rewrite the title. Return it exactly as published."
        ),
        expected_output="List of dicts with title (exact), summary, published, link sorted by date descending",
        agent=scout
    )

    researcher_task = Task(
        description=(
            "You will receive a list of startup funding headlines and summaries from the Scout. "
            "For each entry extract the following fields:\n\n"
            "- company: the EXACT startup name from the headline. "
            "For example if the title is 'Acme AI raises $17M Series A', company is 'Acme AI'. "
            "NEVER use generic descriptions like 'AI startup' or 'mental health platform' as the company name.\n"
            "- amount: funding amount with $ sign, e.g. '$17M'. Use 'Unknown' if not stated.\n"
            "- stage: Seed, Series A, Series B, etc. Use 'Unknown' if not stated.\n"
            "- sector: industry sector, e.g. AI, FinTech, LegalTech, SaaS, HealthTech. Use 'Unknown' if unclear.\n"
            "- investors: names of VC firms or investors mentioned, e.g. 'a16z, YC'. Use 'Not mentioned' if none.\n"
            "- key_people: names and roles of founders, CEOs, CTOs mentioned, e.g. 'Jane Smith (CEO)'. Use 'Not mentioned' if none.\n"
            "- date: published date in YYYY-MM-DD format.\n"
            "- is_startup: true if this is a startup raise, false if it is a VC fund or large enterprise.\n\n"
            "Return only entries where is_startup is true."
        ),
        expected_output=(
            "List of dicts with company (exact name), amount, stage, sector, "
            "investors, key_people, date, title, link"
        ),
        agent=researcher,
        context=[scout_task]
    )

    analyst_task = Task(
        description=(
            "You will receive a list of structured startup entries from the Researcher. "
            "Score each startup 1-10 for PM fit based on this candidate profile:\n"
            "- 7+ years at Amazon, Deloitte, TCS\n"
            "- Strong in B2B SaaS, marketplace, AI-powered products\n"
            "- MBA UC Davis 2026\n"
            "- Targeting Series A and Series B\n"
            "- Low interest in Hardware and pure consumer social\n\n"
            "For each startup return these fields:\n"
            "- company: copy EXACT company name from Researcher output, never use a generic description\n"
            "- amount: copy exactly from Researcher\n"
            "- stage: copy exactly from Researcher\n"
            "- sector: copy exactly from Researcher\n"
            "- investors: copy exactly from Researcher\n"
            "- key_people: copy exactly from Researcher\n"
            "- date: copy exactly from Researcher\n"
            "- fit_score: integer 1-10\n"
            "- reason: one sentence explaining the score\n"
            "- action: one of 'reach out now' (8-10), 'monitor' (5-7), 'skip' (1-4)\n\n"
            "Return ONLY a valid JSON array. No explanation, no markdown, no code blocks. "
            "Sort by date descending, then fit_score descending within the same date.\n"
            "Example:\n"
            '[{"company": "Status AI", "amount": "$17M", "stage": "Series A", "sector": "AI", '
            '"investors": "a16z, YC", "key_people": "Jane Smith (CEO)", "date": "2026-05-20", '
            '"fit_score": 9, "reason": "Strong AI B2B fit at Series A.", "action": "reach out now"}]'
        ),
        expected_output="Valid JSON array sorted by date desc then fit_score desc",
        agent=analyst,
        context=[researcher_task]
    )

    return scout_task, researcher_task, analyst_task


def run_crew(scout_task, researcher_task, analyst_task, scout, researcher, analyst):
    crew = Crew(
        agents=[scout, researcher, analyst],
        tasks=[scout_task, researcher_task, analyst_task],
        process=Process.sequential,
        verbose=False
    )

    result_container = {}

    def kickoff():
        result_container["result"] = crew.kickoff()

    thread = threading.Thread(target=kickoff)
    thread.start()
    thread.join(timeout=300)

    return result_container.get("result", None)


def parse_result(result):
    raw = str(result)
    match = re.search(r'\[.*\]', raw, re.DOTALL)

    if not match:
        return None, raw

    json_str = match.group(0)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
    json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
    json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")

    data = None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        objects = re.findall(r'\{[^{}]+\}', json_str, re.DOTALL)
        data = []
        for obj in objects:
            try:
                parsed = json.loads(obj)
                data.append(parsed)
            except json.JSONDecodeError:
                obj_clean = re.sub(
                    r'("reason"\s*:\s*")(.*?)("(?:\s*,|\s*}))',
                    lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
                    obj,
                    flags=re.DOTALL
                )
                try:
                    parsed = json.loads(obj_clean)
                    data.append(parsed)
                except json.JSONDecodeError:
                    continue

    if not data:
        return None, f"Could not parse any valid entries.\n\nRaw:\n{raw}"

    df_final = pd.DataFrame(data)

    if "date" in df_final.columns:
        df_final["date"] = pd.to_datetime(df_final["date"], errors="coerce")
        df_final = df_final.sort_values(
            ["date", "fit_score"],
            ascending=[False, False]
        ).reset_index(drop=True)
        df_final["date"] = df_final["date"].dt.strftime("%Y-%m-%d")
    else:
        df_final = df_final.sort_values("fit_score", ascending=False).reset_index(drop=True)

    return df_final, None


def export_to_sheets(df_final):
    if not GOOGLE_CREDS:
        return False, "GOOGLE_CREDS secret not set."

    try:
        creds_data = json.loads(GOOGLE_CREDS)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_data, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1

        headers = [
            "run_timestamp", "date", "company", "amount", "stage",
            "sector", "investors", "key_people", "fit_score", "action", "reason"
        ]
        existing = sheet.get_all_values()
        if len(existing) == 0:
            sheet.append_row(headers)

        run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        for _, row in df_final.iterrows():
            sheet.append_row([
                run_time,
                row.get("date", "Unknown"),
                row.get("company", "Unknown"),
                row.get("amount", "Unknown"),
                row.get("stage", "Unknown"),
                row.get("sector", "Unknown"),
                row.get("investors", "Not mentioned"),
                row.get("key_people", "Not mentioned"),
                int(row.get("fit_score", 0)),
                row.get("action", "skip"),
                row.get("reason", "")
            ])

        return True, f"{len(df_final)} rows written at {run_time}"

    except Exception as e:
        return False, str(e)


# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Startup Signal Tracker", page_icon="rocket", layout="wide")
st.title("Startup Signal Tracker")
st.markdown("Multi-agent pipeline: RSS - Extract - Score PM Fit - Google Sheets")

st.sidebar.header("Configuration")
st.sidebar.markdown("**Candidate Profile**")
st.sidebar.markdown("**Abhishek Singh**")
st.sidebar.markdown(
    "[LinkedIn](https://www.linkedin.com/in/abhishek-singh-davis) | "
    "[GitHub](https://github.com/codegeekAbhi) | "
    "[Portfolio](https://www.notion.so/Hi-I-m-Abhishek-Singh-21b6d321e30b804eab8ad37f2783be09)"
)
st.sidebar.markdown("---")
st.sidebar.markdown("- 7+ years: Amazon, Deloitte, TCS")
st.sidebar.markdown("- B2B SaaS, AI, Marketplace")
st.sidebar.markdown("- MBA UC Davis 2026")
st.sidebar.markdown("- Target: Series A/B")
st.sidebar.markdown("- Date range: current month + last month")

run_button = st.sidebar.button("Run Signal Tracker", type="primary")

status = st.empty()
progress = st.progress(0)

if run_button:
    try:
        status.info("Setting up agents...")
        progress.progress(10)

        scout, researcher, analyst = build_agents()
        scout_task, researcher_task, analyst_task = build_tasks(scout, researcher, analyst)

        status.info("Running agents... this takes 2-3 minutes")
        progress.progress(30)

        result = run_crew(scout_task, researcher_task, analyst_task, scout, researcher, analyst)

        if result is None:
            status.error("Crew timed out or failed. Try again.")
            st.stop()

        progress.progress(70)

        df_final, error = parse_result(result)

        if df_final is None:
            status.warning("Could not parse results. Raw output below:")
            st.text(error)
            st.stop()

        status.info("Writing to Google Sheets...")
        progress.progress(85)

        success, msg = export_to_sheets(df_final)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(f"Sheets export skipped: {msg}")

        progress.progress(100)
        status.success(f"Done! {len(df_final)} startups ranked.")

        st.subheader("Ranked Startup List")

        for _, row in df_final.iterrows():
            action = row.get("action", "skip")
            if action == "reach out now":
                icon = "HIGH"
            elif action == "monitor":
                icon = "MED"
            else:
                icon = "LOW"

            label = (
                f"[{icon}] {row.get('date', 'N/A')} - "
                f"{row.get('company', 'Unknown')} - "
                f"Score: {row.get('fit_score', 0)}/10 - "
                f"{action.upper()}"
            )

            with st.expander(label):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Amount", row.get("amount", "Unknown"))
                col2.metric("Stage", row.get("stage", "Unknown"))
                col3.metric("Sector", row.get("sector", "Unknown"))
                col4.metric("Date", row.get("date", "N/A"))

                st.markdown(f"**Investors / VC:** {row.get('investors', 'Not mentioned')}")
                st.markdown(f"**Key People:** {row.get('key_people', 'Not mentioned')}")
                st.markdown(f"**Reason:** {row.get('reason', '')}")

    except Exception as e:
        status.error(f"Error: {str(e)}")
        st.exception(e)
