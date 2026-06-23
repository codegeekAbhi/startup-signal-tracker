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

from datetime import datetime, timezone, timedelta
from google.oauth2.service_account import Credentials
from crewai import Agent, Task, Crew, Process, LLM
import streamlit as st


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
        goal="Extract structured data: company name, funding amount, stage, and sector.",
        backstory="Precise data extractor. Returns clean structured info, uses Unknown when data unavailable.",
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
            "Return list with title, summary, published date, and link. "
            "Sort entries by published date, newest first."
        ),
        expected_output="List of dicts with title, summary, published, link sorted by date descending",
        agent=scout
    )

    researcher_task = Task(
        description=(
            "From the Scout's entries extract: "
            "company, amount, stage, sector, is_startup, date (YYYY-MM-DD format from published field). "
            "Use Unknown when unavailable. Return only startups."
        ),
        expected_output="List of dicts with company, amount, stage, sector, date, title, link",
        agent=researcher,
        context=[scout_task]
    )

    analyst_task = Task(
        description=(
            "Score each startup 1-10 for PM fit. "
            "Provide fit_score as integer, reason as string, action as one of: reach out now, monitor, skip. "
            "Include the date field (YYYY-MM-DD) from the Researcher output for each entry. "
            "Return ONLY a valid JSON array. No explanation, no markdown, no code blocks. "
            "Sort by date descending first, then fit_score descending within the same date.\n"
            'Example format:\n'
            '[{"company": "Acme", "amount": "$10M", "stage": "Series A", "sector": "AI", '
            '"date": "2026-05-20", "fit_score": 9, "reason": "Strong AI fit", "action": "reach out now"}]'
        ),
        expected_output="Valid JSON array sorted by date descending then fit_score descending",
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
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    try:
        data = json.loads(json_str)
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

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}\n\nRaw output:\n{raw}"


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
            "run_timestamp", "date", "company", "amount",
            "stage", "sector", "fit_score", "action", "reason"
        ]
        existing = sheet.get_all_values()
        if len(existing) == 0:
            sheet.append_row(headers)

        run_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        for _, row in df_final.iterrows():
            sheet.append_row([
                run_time,
                row.get("date", "Unknown"),
                row["company"],
                row["amount"],
                row["stage"],
                row["sector"],
                int(row["fit_score"]),
                row["action"],
                row["reason"]
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
            if row["action"] == "reach out now":
                color = "green"
                icon = "HIGH"
            elif row["action"] == "monitor":
                color = "orange"
                icon = "MED"
            else:
                color = "red"
                icon = "LOW"

            label = (
                f"[{icon}] {row.get('date', 'N/A')} - "
                f"{row['company']} - Score: {row['fit_score']}/10 - "
                f"{row['action'].upper()}"
            )

            with st.expander(label):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Amount", row["amount"])
                col2.metric("Stage", row["stage"])
                col3.metric("Sector", row["sector"])
                col4.metric("Date", row.get("date", "N/A"))
                st.markdown(f"**Reason:** {row['reason']}")

    except Exception as e:
        status.error(f"Error: {str(e)}")
        st.exception(e)
