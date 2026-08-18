import streamlit as st
import feedparser
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Startup Signal Tracker",
    page_icon="🚀",
    layout="wide",
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }

    .main {
        background-color: #f0f4ff;
    }

    section[data-testid="stSidebar"] {
        background-color: #1d3a8a;
        color: white;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #ccc;
    }

    .card.green {
        border-left-color: #22c55e;
    }

    .card.yellow {
        border-left-color: #f59e0b;
    }

    .card.red {
        border-left-color: #ef4444;
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    .badge-green {
        background: #dcfce7;
        color: #166534;
    }

    .badge-yellow {
        background: #fef9c3;
        color: #854d0e;
    }

    .badge-red {
        background: #fee2e2;
        color: #991b1b;
    }

    .badge-blue {
        background: #dbeafe;
        color: #1e40af;
    }

    .badge-gray {
        background: #f3f4f6;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# RSS SOURCES
# =============================================================================

RSS_FEEDS = [
    ("TechCrunch Venture", "https://techcrunch.com/category/venture/feed/"),
    ("TechCrunch Startups", "https://techcrunch.com/startups/feed/"),
    ("VentureBeat", "https://venturebeat.com/feed/"),
    ("Crunchbase News", "https://news.crunchbase.com/feed/"),
    ("Sifted", "https://sifted.eu/feed/"),
    ("StrictlyVC", "https://strictlyvc.com/feed/"),
]


STRONG_KEYWORDS = [
    "funding",
    "raises",
    "raised",
    "seed",
    "series a",
    "series b",
    "series c",
    "venture",
    "investment",
    "backed",
    "million",
    "billion",
    "round",
    "valuation",
    "pre-seed",
    "growth round",
    "led by",
    "announces",
]


WEAK_KEYWORDS = [
    "startup",
    "founded",
    "launch",
    "growth",
    "expansion",
    "ai",
    "saas",
    "fintech",
    "healthtech",
    "b2b",
    "platform",
]


BLOCKLIST = [
    "career",
    "job",
    "hiring",
    "podcast",
    "event",
    "webinar",
    "obituary",
    "opinion",
    "review",
    "how to",
    "tutorial",
    "layoff",
    "acqui-hire",
    "bankruptcy",
    "shutdown",
]


DAYS_WINDOW = 7

GROQ_MODEL = "llama-3.3-70b-versatile"


# =============================================================================
# JINA AI ARTICLE READER
# =============================================================================

def fetch_article_content(url: str) -> str:
    """
    Fetch a cleaner article excerpt using Jina Reader.
    No API key required.
    """

    if not url:
        return ""

    try:
        jina_url = f"https://r.jina.ai/{url}"

        headers = {
            "Accept": "text/plain",
            "User-Agent": "StartupSignalTracker/1.0"
        }

        response = requests.get(
            jina_url,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:
            # Give the model a little more context than before.
            return response.text[:2000]

    except Exception as e:
        print(f"Jina error for {url}: {e}")

    return ""


# =============================================================================
# RSS FETCH
# =============================================================================

def fetch_rss_entries():

    cutoff = datetime.utcnow() - timedelta(days=DAYS_WINDOW)

    seen_titles = set()
    entries = []

    for source_name, url in RSS_FEEDS:

        try:
            feed = feedparser.parse(url)

            for entry in feed.entries:

                title = entry.get("title", "").strip()

                if not title:
                    continue

                if title in seen_titles:
                    continue

                title_lower = title.lower()

                # -------------------------------------------------------------
                # Block unwanted articles
                # -------------------------------------------------------------

                if any(word in title_lower for word in BLOCKLIST):
                    continue

                # -------------------------------------------------------------
                # Date filter
                # -------------------------------------------------------------

                published = (
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                )

                if published:

                    pub_dt = datetime(*published[:6])

                    if pub_dt < cutoff:
                        continue

                # -------------------------------------------------------------
                # Funding keyword filter
                # -------------------------------------------------------------

                strong_hit = any(
                    keyword in title_lower
                    for keyword in STRONG_KEYWORDS
                )

                weak_hits = sum(
                    1
                    for keyword in WEAK_KEYWORDS
                    if keyword in title_lower
                )

                if not strong_hit and weak_hits < 2:
                    continue

                seen_titles.add(title)

                entries.append({
                    "title": title,
                    "summary": (entry.get("summary", "") or "")[:500],
                    "link": entry.get("link", ""),
                    "source": source_name,
                    "published": published,
                })

        except Exception as e:
            print(f"RSS error for {source_name}: {e}")
            continue

    # -------------------------------------------------------------------------
    # Sort newest first
    # -------------------------------------------------------------------------

    entries.sort(
        key=lambda x: (
            x["published"]
            if x["published"]
            else (2000, 1, 1, 0, 0, 0)
        ),
        reverse=True
    )

    return entries


# =============================================================================
# GROQ CLIENT
# =============================================================================

@st.cache_resource
def get_groq_client():

    try:

        if "GROQ_API_KEY" not in st.secrets:
            raise ValueError(
                "GROQ_API_KEY is missing from Streamlit secrets."
            )

        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY exists but is empty."
            )

        return Groq(api_key=api_key)

    except Exception as e:
        st.error(f"Groq configuration error: {e}")
        raise


# =============================================================================
# ANALYZE + SCORE STARTUP
# =============================================================================

def analyze_startup(client, entry, article_content=""):

    """
    One Groq call performs BOTH:
    1. Funding information extraction
    2. PM opportunity scoring

    This replaces the previous two-call architecture.
    """

    context = f"""
SOURCE:
{entry.get("source", "Unknown")}

HEADLINE:
{entry.get("title", "")}

RSS SUMMARY:
{entry.get("summary", "")}
"""

    if article_content:

        context += f"""

ARTICLE EXCERPT:
{article_content}
"""

    prompt = f"""
You are analyzing startup funding news for a Product Manager
who is looking for companies worth contacting for PM opportunities.

Analyze the article below.

Your first job is to determine whether the article actually describes
a startup or technology company that recently raised funding.

Then extract the company information and evaluate whether it is a
strong outreach opportunity for a senior Product Manager.

Candidate background:
- Around 8 years across product, engineering and consulting
- Enterprise products
- AI products
- Data platforms
- SaaS
- B2B products
- Technical product management
- AI / ML workflows
- MBA
- Interested in early-stage and growing technology companies

SCORING GUIDE:

9-10:
Excellent outreach target.
Usually Seed to Series B.
Strong AI, SaaS, enterprise software, data, fintech,
healthtech, infrastructure or B2B product relevance.

7-8:
Good PM opportunity.
Growing technology company with reasonable PM relevance.

5-6:
Possible opportunity but weaker signal,
later stage company, unclear PM need, or less relevant sector.

3-4:
Low relevance.

1-2:
Very poor PM target.

ACTION:

Score 8-10:
"reach out now"

Score 5-7:
"monitor"

Score 1-4:
"skip"

IMPORTANT RULES:

1. If this article is NOT actually about a company raising funding,
   set "is_funding_event" to false.

2. Do not invent company names.

3. Do not invent funding amounts.

4. If data is unavailable, use "Unknown".

5. fit_score must be an integer from 1 to 10.

6. action must be exactly one of:
   "reach out now"
   "monitor"
   "skip"

7. Return valid JSON only.

ARTICLE:

{context}

Return exactly this JSON structure:

{{
    "is_funding_event": true,
    "company": "Company Name",
    "amount": "$10M",
    "stage": "Series A",
    "sector": "Enterprise AI",
    "key_people": "Founder or CEO",
    "hq": "San Francisco, US",
    "fit_score": 9,
    "action": "reach out now",
    "reason": "Recently funded enterprise AI startup where technical product experience is strongly relevant."
}}
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured startup funding data. "
                        "Always return valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            # IMPORTANT:
            # Forces valid JSON output.
            response_format={
                "type": "json_object"
            },

            temperature=0.1,

            max_tokens=500,
        )

        raw = response.choices[0].message.content

        if not raw:
            raise ValueError("Groq returned an empty response.")

        result = json.loads(raw)

        if not isinstance(result, dict):
            raise ValueError(
                "Groq response was valid JSON but was not a JSON object."
            )

        # ---------------------------------------------------------------------
        # Normalize / validate fields
        # ---------------------------------------------------------------------

        result.setdefault("is_funding_event", True)
        result.setdefault("company", "Unknown")
        result.setdefault("amount", "Unknown")
        result.setdefault("stage", "Unknown")
        result.setdefault("sector", "Unknown")
        result.setdefault("key_people", "Unknown")
        result.setdefault("hq", "Unknown")
        result.setdefault("fit_score", 5)
        result.setdefault("action", "monitor")
        result.setdefault(
            "reason",
            "Potential startup funding opportunity."
        )

        # ---------------------------------------------------------------------
        # Validate score
        # ---------------------------------------------------------------------

        try:
            score = int(result["fit_score"])
        except (TypeError, ValueError):
            score = 5

        score = max(1, min(10, score))

        result["fit_score"] = score

        # ---------------------------------------------------------------------
        # Normalize action based on score
        # ---------------------------------------------------------------------

        if score >= 8:
            result["action"] = "reach out now"

        elif score >= 5:
            result["action"] = "monitor"

        else:
            result["action"] = "skip"

        # ---------------------------------------------------------------------
        # Clean company
        # ---------------------------------------------------------------------

        company = str(result.get("company", "")).strip()

        if not company:
            result["company"] = "Unknown"

        return result, None

    except Exception as e:

        error_message = str(e)

        return None, error_message


# =============================================================================
# GOOGLE SHEETS EXPORT
# =============================================================================

def export_to_sheets(df):

    try:

        if "GOOGLE_CREDS" not in st.secrets:

            st.warning(
                "GOOGLE_CREDS was not found in Streamlit secrets."
            )

            return False

        creds_dict = json.loads(
            st.secrets["GOOGLE_CREDS"]
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=scopes
        )

        gc = gspread.authorize(creds)

        sh = gc.open(
            "Startup Signal Tracker"
        )

        ws = sh.sheet1

        timestamp = datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M UTC"
        )

        for _, row in df.iterrows():

            ws.append_row([
                timestamp,
                row.get("company", ""),
                row.get("amount", ""),
                row.get("stage", ""),
                row.get("sector", ""),
                row.get("hq", ""),
                row.get("fit_score", ""),
                row.get("action", ""),
                row.get("reason", ""),
                row.get("key_people", ""),
                row.get("source", ""),
                row.get("link", ""),
            ])

        return True

    except Exception as e:

        st.warning(
            f"Google Sheets export failed: {e}"
        )

        return False


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(use_jina=False):

    results = []

    success_count = 0
    failure_count = 0
    non_funding_count = 0
    unknown_company_count = 0

    # -------------------------------------------------------------------------
    # Create Groq client
    # -------------------------------------------------------------------------

    try:
        client = get_groq_client()

    except Exception:
        return []

    # -------------------------------------------------------------------------
    # Status container
    # -------------------------------------------------------------------------

    with st.status(
        "🔍 Scanning RSS feeds...",
        expanded=True
    ) as status:

        entries = fetch_rss_entries()

        st.write(
            f"Found **{len(entries)}** articles matching "
            f"funding signals in the last {DAYS_WINDOW} days."
        )

        if not entries:

            status.update(
                label="No funding articles found.",
                state="error"
            )

            return []

        # ---------------------------------------------------------------------
        # Process articles
        # ---------------------------------------------------------------------

        status.update(
            label="🤖 Analyzing funding signals..."
        )

        progress = st.progress(0)

        error_messages = []

        for i, entry in enumerate(entries):

            article_content = ""

            # -----------------------------------------------------------------
            # Optional Jina article enrichment
            # -----------------------------------------------------------------

            if use_jina and entry.get("link"):

                article_content = fetch_article_content(
                    entry["link"]
                )

                time.sleep(0.2)

            # -----------------------------------------------------------------
            # Groq analysis
            # -----------------------------------------------------------------

            analysis, error = analyze_startup(
                client,
                entry,
                article_content
            )

            # -----------------------------------------------------------------
            # Handle Groq error
            # -----------------------------------------------------------------

            if error:

                failure_count += 1

                error_messages.append({
                    "title": entry.get("title", ""),
                    "source": entry.get("source", ""),
                    "error": error,
                })

                progress.progress(
                    (i + 1) / len(entries)
                )

                # Small pause to avoid hammering API
                time.sleep(0.8)

                continue

            # -----------------------------------------------------------------
            # Reject articles Groq determines aren't funding events
            # -----------------------------------------------------------------

            is_funding = analysis.get(
                "is_funding_event",
                True
            )

            if not is_funding:

                non_funding_count += 1

                progress.progress(
                    (i + 1) / len(entries)
                )

                time.sleep(0.4)

                continue

            # -----------------------------------------------------------------
            # Reject unknown companies
            # -----------------------------------------------------------------

            company = str(
                analysis.get(
                    "company",
                    "Unknown"
                )
            ).strip()

            if (
                not company
                or company.lower() == "unknown"
            ):

                unknown_company_count += 1

                progress.progress(
                    (i + 1) / len(entries)
                )

                time.sleep(0.4)

                continue

            # -----------------------------------------------------------------
            # Successful startup
            # -----------------------------------------------------------------

            success_count += 1

            results.append({
                "company":
                    analysis.get(
                        "company",
                        "Unknown"
                    ),

                "amount":
                    analysis.get(
                        "amount",
                        "Unknown"
                    ),

                "stage":
                    analysis.get(
                        "stage",
                        "Unknown"
                    ),

                "sector":
                    analysis.get(
                        "sector",
                        "Unknown"
                    ),

                "hq":
                    analysis.get(
                        "hq",
                        "Unknown"
                    ),

                "key_people":
                    analysis.get(
                        "key_people",
                        "Unknown"
                    ),

                "fit_score":
                    analysis.get(
                        "fit_score",
                        5
                    ),

                "action":
                    analysis.get(
                        "action",
                        "monitor"
                    ),

                "reason":
                    analysis.get(
                        "reason",
                        ""
                    ),

                "source":
                    entry.get(
                        "source",
                        ""
                    ),

                "link":
                    entry.get(
                        "link",
                        ""
                    ),

                "title":
                    entry.get(
                        "title",
                        ""
                    ),
            })

            progress.progress(
                (i + 1) / len(entries)
            )

            # -----------------------------------------------------------------
            # Throttle slightly
            # -----------------------------------------------------------------

            time.sleep(0.6)

        # ---------------------------------------------------------------------
        # Pipeline summary
        # ---------------------------------------------------------------------

        st.write("---")

        st.write(
            f"✅ Successfully analyzed: **{success_count}**"
        )

        st.write(
            f"⚠️ API / parsing failures: **{failure_count}**"
        )

        st.write(
            f"📰 Not actual funding events: **{non_funding_count}**"
        )

        st.write(
            f"❓ Company could not be identified: "
            f"**{unknown_company_count}**"
        )

        # ---------------------------------------------------------------------
        # SHOW ERRORS
        # ---------------------------------------------------------------------

        if error_messages:
        
            st.warning(
                f"Groq encountered {len(error_messages)} error(s). "
                "Error details are shown below."
            )
        
            for error_item in error_messages:
        
                st.markdown(
                    f"**Article:** {error_item['title']}"
                )
        
                st.caption(
                    f"Source: {error_item['source']}"
                )
        
                st.code(
                    error_item["error"]
                )
        
                st.markdown("---")

        # ---------------------------------------------------------------------
        # Final status
        # ---------------------------------------------------------------------

        if results:

            status.update(
                label=(
                    f"Done — "
                    f"{len(results)} startups scored"
                ),
                state="complete"
            )

        elif failure_count > 0:

            status.update(
                label=(
                    "Pipeline finished but Groq "
                    "returned errors."
                ),
                state="error"
            )

        else:

            status.update(
                label=(
                    "Pipeline finished — no valid "
                    "funding startups identified."
                ),
                state="complete"
            )

    # -------------------------------------------------------------------------
    # Sort highest score first
    # -------------------------------------------------------------------------

    return sorted(
        results,
        key=lambda x: x["fit_score"],
        reverse=True
    )


# =============================================================================
# UI HELPERS
# =============================================================================

def action_badge(action):

    action = (
        action or ""
    ).lower()

    if "reach" in action:

        return (
            '<span class="badge badge-green">'
            '🟢 Reach Out Now'
            '</span>',
            "green"
        )

    elif "monitor" in action:

        return (
            '<span class="badge badge-yellow">'
            '🟡 Monitor'
            '</span>',
            "yellow"
        )

    else:

        return (
            '<span class="badge badge-red">'
            '🔴 Skip'
            '</span>',
            "red"
        )


def render_card(result):

    badge_html, color = action_badge(
        result.get("action", "")
    )

    score = result.get(
        "fit_score",
        5
    )

    hq = result.get(
        "hq",
        "Unknown"
    )

    source = result.get(
        "source",
        ""
    )

    company = result.get(
        "company",
        "Unknown"
    )

    amount = result.get(
        "amount",
        "Unknown"
    )

    stage = result.get(
        "stage",
        "Unknown"
    )

    sector = result.get(
        "sector",
        "Unknown"
    )

    key_people = result.get(
        "key_people",
        "Unknown"
    )

    reason = result.get(
        "reason",
        ""
    )

    link = result.get(
        "link",
        ""
    )

    title = result.get(
        "title",
        ""
    )

    st.markdown(
        f"""
        <div class="card {color}">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                flex-wrap:wrap;
                gap:8px;
            ">

                <h3 style="
                    margin:0;
                    font-size:1.1rem;
                ">
                    {company}
                </h3>

                {badge_html}

                <span class="badge badge-blue">
                    Fit Score: {score}/10
                </span>

            </div>

            <div style="
                margin-top:0.6rem;
                display:flex;
                flex-wrap:wrap;
                gap:6px;
            ">

                <span class="badge badge-gray">
                    💰 {amount}
                </span>

                <span class="badge badge-gray">
                    📊 {stage}
                </span>

                <span class="badge badge-gray">
                    🏭 {sector}
                </span>

                <span class="badge badge-gray">
                    📍 {hq}
                </span>

                <span class="badge badge-gray">
                    👤 {key_people}
                </span>

                <span class="badge badge-gray">
                    📰 {source}
                </span>

            </div>

            <p style="
                margin:0.6rem 0 0.3rem;
                font-size:0.9rem;
                color:#374151;
            ">
                {reason}
            </p>

            <a
                href="{link}"
                target="_blank"
                style="
                    font-size:0.8rem;
                    color:#1d3a8a;
                "
            >
                🔗 {title[:90]}
            </a>

        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "## Abhishek Singh"
    )

    st.markdown(
        "Senior PM | AI Products | MBA UC Davis 2026"
    )

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/aabhishek-singh/) "
        "| "
        "[Portfolio](https://abhishek-s-portfolio--abhisheksngh180.replit.app/)"
    )

    st.markdown("---")

    st.markdown(
        "## 🚀 Startup Signal Tracker"
    )

    st.markdown(
        "Monitors funding news and ranks "
        "startups by PM fit."
    )

    st.markdown("---")

    st.markdown(
        "**Settings**"
    )

    use_jina = st.toggle(
        "Deep article scan (Jina AI)",
        value=False,
        help=(
            "Fetches article content for richer "
            "startup extraction. Slower but "
            "usually more accurate."
        )
    )

    st.markdown("---")

    st.markdown(
        "**Filters**"
    )

    filter_action = st.multiselect(
        "Action",
        [
            "reach out now",
            "monitor",
            "skip"
        ],
        default=[
            "reach out now",
            "monitor"
        ],
    )

    filter_stage = st.multiselect(
        "Stage",
        [
            "Pre-seed",
            "Seed",
            "Series A",
            "Series B",
            "Series C",
            "Growth",
            "Unknown",
        ],
        default=[],
    )

    filter_source = st.multiselect(
        "Source",
        [
            name
            for name, _ in RSS_FEEDS
        ],
        default=[],
    )

    st.markdown("---")

    st.markdown(
        "**Sources**"
    )

    for name, url in RSS_FEEDS:

        domain = (
            url
            .split("/")[2]
            .replace("www.", "")
        )

        st.markdown(
            f"• {name} ({domain})"
        )


# =============================================================================
# MAIN PAGE
# =============================================================================

st.markdown(
    "# 🚀 Startup Signal Tracker"
)

st.markdown(
    f"Live funding signals from "
    f"{len(RSS_FEEDS)} sources, "
    f"last {DAYS_WINDOW} days only"
)


# =============================================================================
# BUTTONS
# =============================================================================

col1, col2, col3 = st.columns(
    [1, 1, 2]
)

with col1:

    run_btn = st.button(
        "▶ Run Pipeline",
        type="primary",
        use_container_width=True
    )

with col2:

    sheets_btn = st.button(
        "📊 Export to Sheets",
        use_container_width=True
    )


# =============================================================================
# RUN PIPELINE
# =============================================================================

if run_btn:

    results = run_pipeline(
        use_jina=use_jina
    )

    st.session_state[
        "results"
    ] = results

    st.session_state[
        "run_time"
    ] = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    if results:

        reach = sum(
            1
            for result in results
            if "reach" in result["action"].lower()
        )

        monitor = sum(
            1
            for result in results
            if "monitor" in result["action"].lower()
        )

        skip = sum(
            1
            for result in results
            if "skip" in result["action"].lower()
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Total Startups",
            len(results)
        )

        m2.metric(
            "🟢 Reach Out Now",
            reach
        )

        m3.metric(
            "🟡 Monitor",
            monitor
        )

        m4.metric(
            "🔴 Skip",
            skip
        )


# =============================================================================
# GET SESSION RESULTS
# =============================================================================

results = st.session_state.get(
    "results",
    []
)

run_time = st.session_state.get(
    "run_time",
    ""
)


# =============================================================================
# GOOGLE SHEETS
# =============================================================================

if sheets_btn:

    if not results:

        st.warning(
            "Run the pipeline before exporting."
        )

    else:

        df = pd.DataFrame(
            results
        )

        success = export_to_sheets(
            df
        )

        if success:

            st.success(
                "Exported to Google Sheets."
            )


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if results:

    if run_time:

        st.caption(
            f"Last run: {run_time}"
        )

    # -------------------------------------------------------------------------
    # Apply filters
    # -------------------------------------------------------------------------

    filtered = results

    if filter_action:

        filtered = [
            result
            for result in filtered
            if any(
                action
                in result["action"].lower()
                for action in filter_action
            )
        ]

    if filter_stage:

        filtered = [
            result
            for result in filtered
            if result["stage"] in filter_stage
        ]

    if filter_source:

        filtered = [
            result
            for result in filtered
            if result["source"] in filter_source
        ]

    # -------------------------------------------------------------------------
    # Result count
    # -------------------------------------------------------------------------

    st.markdown(
        f"### Showing {len(filtered)} startups"
    )

    # -------------------------------------------------------------------------
    # Tabs
    # -------------------------------------------------------------------------

    tab1, tab2 = st.tabs([
        "📋 Cards",
        "📊 Table"
    ])

    with tab1:

        if not filtered:

            st.info(
                "No startups match your current filters."
            )

        else:

            for result in filtered:

                render_card(
                    result
                )

    with tab2:

        if filtered:

            df = pd.DataFrame(
                filtered
            )

            display_cols = [
                "company",
                "amount",
                "stage",
                "sector",
                "hq",
                "fit_score",
                "action",
                "reason",
                "source",
                "link",
            ]

            available_cols = [
                column
                for column in display_cols
                if column in df.columns
            ]

            st.dataframe(
                df[available_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "link": st.column_config.LinkColumn(
                        "Article"
                    ),
                    "fit_score": st.column_config.NumberColumn(
                        "Fit Score",
                        min_value=1,
                        max_value=10,
                        format="%d/10"
                    ),
                }
            )

else:

    st.info(
        "Click **Run Pipeline** to scan "
        "for funding signals."
    )
