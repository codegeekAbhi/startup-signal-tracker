import streamlit as st
import feedparser
import json
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Startup Signal Tracker",
    page_icon="🚀",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }
    .main { background-color: #f0f4ff; }
    section[data-testid="stSidebar"] { background-color: #1d3a8a; color: white; }
    section[data-testid="stSidebar"] * { color: white !important; }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #ccc;
    }
    .card.green  { border-left-color: #22c55e; }
    .card.yellow { border-left-color: #f59e0b; }
    .card.red    { border-left-color: #ef4444; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-green  { background: #dcfce7; color: #166534; }
    .badge-yellow { background: #fef9c3; color: #854d0e; }
    .badge-red    { background: #fee2e2; color: #991b1b; }
    .badge-blue   { background: #dbeafe; color: #1e40af; }
    .badge-gray   { background: #f3f4f6; color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://techcrunch.com/category/venture/feed/",
    "https://techcrunch.com/startups/feed/",
    "https://venturebeat.com/feed/",
]

STRONG_KEYWORDS = [
    "funding", "raises", "seed", "series a", "series b", "series c",
    "venture", "investment", "backed", "million", "billion", "round",
]
WEAK_KEYWORDS = [
    "startup", "founded", "launch", "growth", "expansion",
    "AI", "SaaS", "fintech", "healthtech", "B2B",
]
BLOCKLIST = [
    "career", "job", "hiring", "podcast", "event", "webinar",
    "obituary", "opinion", "review", "how to", "tutorial",
]

DAYS_WINDOW = 7

# ── RSS Fetch ─────────────────────────────────────────────────────────────────────
def fetch_rss_entries():
    cutoff = datetime.utcnow() - timedelta(days=DAYS_WINDOW)
    seen_titles = set()
    entries = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                title = e.get("title", "").strip()
                if not title or title in seen_titles:
                    continue

                title_lower = title.lower()
                if any(b in title_lower for b in BLOCKLIST):
                    continue

                published = e.get("published_parsed") or e.get("updated_parsed")
                if published:
                    pub_dt = datetime(*published[:6])
                    if pub_dt < cutoff:
                        continue

                strong_hit = any(k in title_lower for k in STRONG_KEYWORDS)
                weak_hits  = sum(1 for k in WEAK_KEYWORDS if k in title_lower)
                if not strong_hit and weak_hits < 2:
                    continue

                seen_titles.add(title)
                entries.append({
                    "title":   title,
                    "summary": (e.get("summary", "") or "")[:300],
                    "link":    e.get("link", ""),
                })
        except Exception:
            continue

    return entries

# ── Groq client ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Step 1: Extract startup info ─────────────────────────────────────────────────
def extract_startup_info(client, entry):
    prompt = f"""Extract startup funding info as JSON only, no markdown:
Title: {entry['title']}
Summary: {entry['summary'][:150]}
{{"company":"name or Unknown","amount":"$X or Unknown","stage":"Seed/Series A/B/C/Unknown","sector":"sector","key_people":"name or Unknown"}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        return json.loads(raw)
    except Exception as e:
        st.error(f"EXTRACTION ERROR for '{entry['title'][:50]}...': {repr(e)}")
        return {
            "company": "Unknown",
            "amount": "Unknown",
            "stage": "Unknown",
            "sector": "Unknown",
            "key_people": "Unknown",
        }

# ── Step 2: Score PM fit ──────────────────────────────────────────────────────────
def score_pm_fit(client, info, entry):
    prompt = f"""You are a PM job seeker evaluating startup funding news for outreach opportunities.
Score this startup's fit for a Senior PM role. Consider: stage (earlier = more opportunity), 
AI/SaaS/B2B sectors score higher, larger rounds mean more hiring budget.

Startup: {info['company']}
Amount: {info['amount']}
Stage: {info['stage']}
Sector: {info['sector']}
Title: {entry['title']}

Return ONLY valid JSON. No markdown, no explanation.
{{
  "fit_score": <integer 1-10>,
  "action": "reach out now / monitor / skip",
  "reason": "one sentence explanation"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        return json.loads(raw)
    except Exception as e:
        st.error(f"SCORING ERROR for '{info.get('company','?')}': {repr(e)}")
        return {
            "fit_score": 5,
            "action": "monitor",
            "reason": "Could not score automatically.",
        }

# ── Google Sheets export ──────────────────────────────────────────────────────────
def export_to_sheets(df):
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDS"])
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open("Startup Signal Tracker")
        ws = sh.sheet1

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        for _, row in df.iterrows():
            ws.append_row([
                timestamp,
                row.get("company", ""),
                row.get("amount", ""),
                row.get("stage", ""),
                row.get("sector", ""),
                row.get("fit_score", ""),
                row.get("action", ""),
                row.get("reason", ""),
                row.get("key_people", ""),
                row.get("link", ""),
            ])
        return True
    except Exception as e:
        st.warning(f"Google Sheets export failed: {e}")
        return False

# ── Run pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline():
    client = get_groq_client()
    results = []

    with st.status("🔍 Scanning RSS feeds...", expanded=True) as status:
        entries = fetch_rss_entries()
        st.write(f"Found **{len(entries)}** articles matching funding signals")

        if not entries:
            status.update(label="No articles found. Try again later.", state="error")
            return []

        status.update(label="🤖 Extracting and scoring startups...")

        progress = st.progress(0)
        for i, entry in enumerate(entries):
            info = extract_startup_info(client, entry)
            time.sleep(0.5)

            if info.get("company", "Unknown") == "Unknown":
                progress.progress((i + 1) / len(entries))
                continue

            score = score_pm_fit(client, info, entry)
            time.sleep(0.5)

            results.append({
                "company":    info.get("company", "Unknown"),
                "amount":     info.get("amount", "Unknown"),
                "stage":      info.get("stage", "Unknown"),
                "sector":     info.get("sector", "Unknown"),
                "key_people": info.get("key_people", "Unknown"),
                "fit_score":  score.get("fit_score", 5),
                "action":     score.get("action", "monitor"),
                "reason":     score.get("reason", ""),
                "link":       entry.get("link", ""),
                "title":      entry.get("title", ""),
            })
            progress.progress((i + 1) / len(entries))

        status.update(label=f"✅ Done — {len(results)} startups scored", state="complete")

    return sorted(results, key=lambda x: x["fit_score"], reverse=True)

# ── UI ─────────────────────────────────────────────────────────────────────────────
def action_badge(action):
    action = (action or "").lower()
    if "reach" in action:
        return '<span class="badge badge-green">🟢 Reach Out Now</span>', "green"
    elif "monitor" in action:
        return '<span class="badge badge-yellow">🟡 Monitor</span>', "yellow"
    else:
        return '<span class="badge badge-red">🔴 Skip</span>', "red"

def render_card(r):
    badge_html, color = action_badge(r["action"])
    score = r.get("fit_score", 5)
    st.markdown(f"""
    <div class="card {color}">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <h3 style="margin:0; font-size:1.1rem;">{r['company']}</h3>
            {badge_html}
            <span class="badge badge-blue">Fit Score: {score}/10</span>
        </div>
        <div style="margin-top:0.6rem; display:flex; flex-wrap:wrap; gap:6px;">
            <span class="badge badge-gray">💰 {r['amount']}</span>
            <span class="badge badge-gray">📊 {r['stage']}</span>
            <span class="badge badge-gray">🏭 {r['sector']}</span>
            <span class="badge badge-gray">👤 {r['key_people']}</span>
        </div>
        <p style="margin:0.6rem 0 0.3rem; font-size:0.9rem; color:#374151;">{r['reason']}</p>
        <a href="{r['link']}" target="_blank" style="font-size:0.8rem; color:#1d3a8a;">📰 {r['title'][:80]}...</a>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Startup Signal Tracker")
    st.markdown("Monitors funding news and ranks startups by PM fit.")
    st.markdown("---")
    st.markdown("**Filters**")
    filter_action = st.multiselect(
        "Action",
        ["reach out now", "monitor", "skip"],
        default=["reach out now", "monitor"],
    )
    filter_stage = st.multiselect(
        "Stage",
        ["Seed", "Series A", "Series B", "Series C", "Growth", "Unknown"],
        default=[],
    )
    st.markdown("---")
    st.markdown("**Sources**")
    for f in RSS_FEEDS:
        domain = f.split("/")[2].replace("www.", "")
        st.markdown(f"• {domain}")

# ── Main ───────────────────────────────────────────────────────────────────────────
st.markdown("# 🚀 Startup Signal Tracker")
st.markdown("Funding signals → PM fit scores → ranked outreach list")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)
with col2:
    sheets_btn = st.button("📊 Export to Sheets", use_container_width=True)

if run_btn:
    results = run_pipeline()
    st.session_state["results"] = results

    if results:
        reach = sum(1 for r in results if "reach" in r["action"].lower())
        monitor = sum(1 for r in results if "monitor" in r["action"].lower())
        skip = sum(1 for r in results if "skip" in r["action"].lower())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Startups", len(results))
        m2.metric("🟢 Reach Out Now", reach)
        m3.metric("🟡 Monitor", monitor)
        m4.metric("🔴 Skip", skip)

results = st.session_state.get("results", [])

if sheets_btn and results:
    df = pd.DataFrame(results)
    ok = export_to_sheets(df)
    if ok:
        st.success("✅ Exported to Google Sheets")

if results:
    filtered = results
    if filter_action:
        filtered = [r for r in filtered if any(a in r["action"].lower() for a in filter_action)]
    if filter_stage:
        filtered = [r for r in filtered if r["stage"] in filter_stage]

    st.markdown(f"### Showing {len(filtered)} startups")

    tab1, tab2 = st.tabs(["📋 Cards", "📊 Table"])

    with tab1:
        for r in filtered:
            render_card(r)

    with tab2:
        df = pd.DataFrame(filtered)
        display_cols = ["company", "amount", "stage", "sector", "fit_score", "action", "reason"]
        st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True)

else:
    st.info("Click **Run Pipeline** to scan for funding signals.")
