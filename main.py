# Startup Signal Tracker — Streamlit UI
import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew, Process, LLM
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import re

os.environ["LITELLM_CACHE"] = "False"
os.environ["GROQ_CACHE"] = "False"

st.set_page_config(page_title="Startup Signal Tracker", page_icon="🚀", layout="wide")
st.title("🚀 Startup Signal Tracker")
st.markdown("Multi-agent pipeline: RSS → Extract → Score PM Fit → Google Sheets")

st.sidebar.header("Configuration")
st.sidebar.markdown("**Candidate Profile**")
st.sidebar.markdown("- 7+ years: Amazon, Deloitte, TCS")
st.sidebar.markdown("- B2B SaaS, AI, Marketplace")
st.sidebar.markdown("- MBA UC Davis 2026")
st.sidebar.markdown("- Target: Series A/B")

run_button = st.sidebar.button("🔍 Run Signal Tracker", type="primary")

status = st.empty()
progress = st.progress(0)

def fetch_rss_entries():
    feeds = [
        "https://techcrunch.com/feed/",
        "https://techcrunch.com/category/startups/feed/",
        "https://venturebeat.com/feed/"
    ]
    keywords = ['raises', 'funding', 'million', 'series', 'seed', 'venture', 'backed', 'investment']
    noise = ['layoffs', 'acqui', 'ipo', 'public offering', 'bankrupt']
    cutoff = datetime.utcnow() - timedelta(days=7)

    seen, entries = set(), []
    for url in feeds:
        feed = feedparser.parse(url)
        for e in feed.entries:
            title = e.get('title', '')
            tl = title.lower()
            if title in seen:
                continue
            if not any(k in tl for k in keywords):
                continue
            if any(n in tl for n in noise):
                continue
            pub = e.get('published_parsed') or e.get('updated_parsed')
            if pub:
                pub_dt = datetime(*pub[:6])
                if pub_dt < cutoff:
                    continue
            seen.add(title)
            entries.append({
                'title': title,
                'summary': e.get('summary', '')[:300],
                'published': e.get('published', 'Unknown'),
                'link': e.get('link', '')
            })
    return entries

if run_button:
    try:
        status.info("📡 Fetching RSS feeds...")
        progress.progress(10)

        entries = fetch_rss_entries()

        if not entries:
            status.warning("⚠️ No funding articles found in the last 7 days.")
            st.stop()

        status.info(f"✅ Found {len(entries)} articles. Setting up agents...")
        progress.progress(25)

        groq_key = os.environ.get("GROQ_API_KEY", "")

        llm = LLM(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )

        researcher = Agent(
            role="Startup Data Researcher",
            goal="Extract structured data: company name, funding amount, stage, and sector.",
            backstory="Precise data extractor. Returns clean structured info, uses 'Unknown' when data unavailable.",
            llm=llm, verbose=False, allow_delegation=False
        )

        analyst = Agent(
            role="PM Fit Analyst",
            goal="Score each startup 1-10 for PM fit and recommend: reach out now, monitor, or skip.",
            backstory="""Evaluates startups for a senior PM candidate: 7+ years at Amazon/Deloitte/TCS,
            B2B SaaS and AI experience, MBA UC Davis 2026, targeting Series A/B. Low fit for Hardware and consumer social.""",
            llm=llm, verbose=False, allow_delegation=False
        )

        entries_text = "\n\n".join([
            f"Title: {e['title']}\nSummary: {e['summary']}\nLink: {e['link']}"
            for e in entries[:15]
        ])

        researcher_task = Task(
            description=f"""Extract structured data from these real startup funding articles:

{entries_text}

For each article extract:
- company: the startup's actual name from the title (NEVER use 'Unknown' or placeholder names)
- amount: funding amount (e.g. '$17M') or 'Unknown'
- stage: funding stage (e.g. 'Series A') or 'Unknown'
- sector: industry sector (e.g. 'AI', 'FinTech', 'SaaS') or 'Unknown'
- title: original title
- link: original link
- key_people: any founder/CEO/CTO names mentioned, or 'Not found'

Return only real startups (skip big public companies). Use 'Unknown' only for missing fields, never for company name.""",
            expected_output="List of dicts with company, amount, stage, sector, title, link, key_people",
            agent=researcher
        )

        analyst_task = Task(
            description="""Score each startup from the Researcher for PM fit 1-10.

Candidate profile: 7+ years Amazon/Deloitte/TCS, B2B SaaS & AI focus, MBA UC Davis 2026, targeting Series A/B.
Low fit: Hardware, pure consumer social.

For each startup return:
- company: exact name from Researcher (never change it)
- amount, stage, sector: carry forward exactly
- fit_score: integer 1-10
- reason: one sentence
- action: 'reach out now' (8-10), 'monitor' (5-7), 'skip' (1-4)
- key_people: carry forward from Researcher

Return ONLY a valid JSON array, no markdown, no code blocks, sorted by fit_score descending.
Example: [{"company": "Acme AI", "amount": "$10M", "stage": "Series A", "sector": "AI", "fit_score": 9, "reason": "Strong AI B2B fit", "action": "reach out now", "key_people": "Jane Smith (CEO)"}]""",
            expected_output="Valid JSON array sorted by fit_score descending",
            agent=analyst,
            context=[researcher_task]
        )

        status.info("🤖 Running agents... this takes 2-3 minutes")
        progress.progress(40)

        crew = Crew(
            agents=[researcher, analyst],
            tasks=[researcher_task, analyst_task],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        progress.progress(75)

        raw = str(result)
        match = re.search(r'\[.*\]', raw, re.DOTALL)

        if match:
            json_str = match.group(0)
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = ' '.join(json_str.split())

            try:
                data = json.loads(json_str)
                df = pd.DataFrame(data).sort_values('fit_score', ascending=False).reset_index(drop=True)

                status.info("📊 Writing to Google Sheets...")
                progress.progress(88)

                creds_json = os.environ.get("GOOGLE_CREDS")
                if creds_json:
                    creds_data = json.loads(creds_json)
                    scopes = ["https://www.googleapis.com/auth/spreadsheets",
                              "https://www.googleapis.com/auth/drive"]
                    creds = Credentials.from_service_account_info(creds_data, scopes=scopes)
                    client = gspread.authorize(creds)
                    sheet = client.open("Startup Signal Tracker").sheet1

                    headers = ['run_timestamp', 'company', 'amount', 'stage', 'sector', 'fit_score', 'action', 'reason', 'key_people']
                    existing = sheet.get_all_values()
                    if len(existing) == 0:
                        sheet.append_row(headers)

                    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                    for _, row in df.iterrows():
                        sheet.append_row([
                            run_time,
                            row.get('company', 'Unknown'),
                            row.get('amount', 'Unknown'),
                            row.get('stage', 'Unknown'),
                            row.get('sector', 'Unknown'),
                            int(row.get('fit_score', 0)),
                            row.get('action', 'skip'),
                            row.get('reason', ''),
                            row.get('key_people', 'Not found')
                        ])

                progress.progress(100)
                status.success(f"✅ Done! {len(df)} startups ranked.")
                st.subheader("📋 Ranked Startup List")

                for _, row in df.iterrows():
                    action = row.get('action', 'skip')
                    if action == 'reach out now':
                        color = "🟢"
                    elif action == 'monitor':
                        color = "🟡"
                    else:
                        color = "🔴"

                    company = row.get('company', 'Unknown')
                    score = row.get('fit_score', 0)

                    with st.expander(f"{color} {company} — Score: {score}/10 — {action.upper()}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Amount", row.get('amount', 'Unknown'))
                        col2.metric("Stage", row.get('stage', 'Unknown'))
                        col3.metric("Sector", row.get('sector', 'Unknown'))
                        st.markdown(f"**Reason:** {row.get('reason', '')}")
                        st.markdown(f"**Key People:** {row.get('key_people', 'Not found')}")
                        if row.get('link'):
                            st.markdown(f"[🔗 Read article]({row.get('link')})")

            except json.JSONDecodeError:
                status.warning("⚠️ Could not parse JSON. Raw output below:")
                st.text(raw)
        else:
            status.warning("⚠️ No results found. Raw output below:")
            st.text(raw)

    except Exception as e:
        status.error(f"❌ Error: {str(e)}")
        st.exception(e)
