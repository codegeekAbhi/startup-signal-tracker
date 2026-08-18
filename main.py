import streamlit as st
import feedparser
import json
import time
import re
import requests
import pandas as pd

from datetime import datetime, timedelta, timezone
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

st.markdown(
    """
    <style>
        html, body, [class*="css"] {
            font-family: Arial, sans-serif;
        }

        section[data-testid="stSidebar"] {
            background-color: #24428f;
        }

        section[data-testid="stSidebar"] * {
            color: white;
        }

        .startup-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .startup-title {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .startup-meta {
            font-size: 14px;
            color: #4b5563;
            margin-bottom: 7px;
        }

        .score {
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONFIG
# =============================================================================

DAYS_WINDOW = 7


RSS_FEEDS = [
    (
        "TechCrunch Venture",
        "https://techcrunch.com/category/venture/feed/",
    ),
    (
        "TechCrunch Startups",
        "https://techcrunch.com/startups/feed/",
    ),
    (
        "VentureBeat",
        "https://venturebeat.com/feed/",
    ),
    (
        "Crunchbase News",
        "https://news.crunchbase.com/feed/",
    ),
    (
        "Sifted",
        "https://sifted.eu/feed/",
    ),
    (
        "StrictlyVC",
        "https://strictlyvc.com/feed/",
    ),
]


STRONG_KEYWORDS = [
    "funding",
    "raises",
    "raised",
    "raise",
    "seed",
    "pre-seed",
    "series a",
    "series b",
    "series c",
    "series d",
    "venture",
    "investment",
    "backed",
    "million",
    "billion",
    "round",
    "valuation",
    "fund",
    "led by",
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
    "software",
]


BLOCKLIST = [
    "career advice",
    "podcast",
    "webinar",
    "obituary",
    "tutorial",
    "how to",
    "layoff",
    "bankruptcy",
    "shutdown",
]


# =============================================================================
# GROQ
# =============================================================================

@st.cache_resource
def get_groq_client():
    """
    Creates Groq client from Streamlit secrets.
    """

    if "GROQ_API_KEY" not in st.secrets:
        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit secrets."
        )

    api_key = st.secrets["GROQ_API_KEY"]

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY exists but is empty."
        )

    return Groq(
        api_key=api_key
    )


@st.cache_data(ttl=600)
def fetch_available_model_ids():
    """
    Ask Groq which models are available to this API key.

    Cached for 10 minutes.
    """

    try:
        client = get_groq_client()

        response = client.models.list()

        model_ids = []

        for model in response.data:
            model_id = getattr(
                model,
                "id",
                None
            )

            if model_id:
                model_ids.append(
                    model_id
                )

        return sorted(
            list(set(model_ids))
        ), None

    except Exception as e:
        return [], str(e)


def is_likely_text_model(model_id):
    """
    Tries to exclude obviously unsuitable speech,
    moderation, embedding, and TTS models.
    """

    model_lower = model_id.lower()

    excluded_terms = [
        "whisper",
        "speech",
        "tts",
        "embedding",
        "moderation",
        "guard",
    ]

    return not any(
        term in model_lower
        for term in excluded_terms
    )


def choose_default_model(model_ids):
    """
    Pick a preferred model only if Groq says
    it is actually available to this key.
    """

    if not model_ids:
        return None

    preferred_models = [
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
    ]

    for preferred in preferred_models:

        if preferred in model_ids:
            return preferred

    text_models = [
        model_id
        for model_id in model_ids
        if is_likely_text_model(
            model_id
        )
    ]

    if text_models:
        return text_models[0]

    return model_ids[0]


# =============================================================================
# ARTICLE FETCHING
# =============================================================================

def fetch_article_content(url):
    """
    Optional deep article fetch through Jina Reader.
    """

    if not url:
        return ""

    try:

        jina_url = (
            f"https://r.jina.ai/{url}"
        )

        response = requests.get(
            jina_url,
            headers={
                "Accept": "text/plain",
                "User-Agent":
                    "StartupSignalTracker/1.0",
            },
            timeout=15,
        )

        if response.status_code == 200:

            return response.text[:4000]

    except Exception:
        pass

    return ""


# =============================================================================
# RSS
# =============================================================================

def get_entry_date(entry):

    published = (
        entry.get("published_parsed")
        or entry.get("updated_parsed")
    )

    if not published:
        return None

    try:

        return datetime(
            *published[:6],
            tzinfo=timezone.utc,
        )

    except Exception:

        return None


def fetch_rss_entries():

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=DAYS_WINDOW)
    )

    results = []

    seen_links = set()
    seen_titles = set()

    for source_name, feed_url in RSS_FEEDS:

        try:

            feed = feedparser.parse(
                feed_url
            )

            for entry in feed.entries:

                title = (
                    entry
                    .get("title", "")
                    .strip()
                )

                link = (
                    entry
                    .get("link", "")
                    .strip()
                )

                summary = (
                    entry
                    .get("summary", "")
                    or ""
                )

                if not title:
                    continue

                title_lower = (
                    title.lower()
                )

                if title_lower in seen_titles:
                    continue

                if (
                    link
                    and link in seen_links
                ):
                    continue

                if any(
                    blocked
                    in title_lower
                    for blocked
                    in BLOCKLIST
                ):
                    continue

                article_date = (
                    get_entry_date(entry)
                )

                if (
                    article_date
                    and article_date < cutoff
                ):
                    continue

                searchable_text = (
                    title_lower
                    + " "
                    + summary.lower()
                )

                strong_hit = any(
                    keyword
                    in searchable_text
                    for keyword
                    in STRONG_KEYWORDS
                )

                weak_hits = sum(
                    1
                    for keyword
                    in WEAK_KEYWORDS
                    if keyword
                    in searchable_text
                )

                if (
                    not strong_hit
                    and weak_hits < 2
                ):
                    continue

                seen_titles.add(
                    title_lower
                )

                if link:
                    seen_links.add(
                        link
                    )

                results.append(
                    {
                        "title": title,
                        "summary":
                            summary[:1200],
                        "link": link,
                        "source":
                            source_name,
                        "published":
                            article_date,
                    }
                )

        except Exception as e:

            print(
                f"RSS error for "
                f"{source_name}: {e}"
            )

    results.sort(
        key=lambda item:
            item["published"]
            or datetime(
                2000,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        reverse=True,
    )

    return results


# =============================================================================
# JSON PARSER
# =============================================================================

def parse_json_response(raw_text):
    """
    Handles:
    - pure JSON
    - ```json ... ```
    - extra text surrounding JSON
    """

    if not raw_text:

        raise ValueError(
            "Model returned an empty response."
        )

    text = raw_text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # First attempt
    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:
        pass

    # Find first JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if (
        first_brace == -1
        or last_brace == -1
        or last_brace <= first_brace
    ):

        raise ValueError(
            "Could not find a JSON object "
            "in the model response."
        )

    possible_json = text[
        first_brace:
        last_brace + 1
    ]

    return json.loads(
        possible_json
    )


# =============================================================================
# STARTUP ANALYSIS
# =============================================================================

def analyze_startup(
    client,
    model_id,
    entry,
    article_content="",
):

    article_text = f"""
SOURCE:
{entry.get("source", "Unknown")}

HEADLINE:
{entry.get("title", "")}

RSS SUMMARY:
{entry.get("summary", "")}
"""

    if article_content:

        article_text += f"""

ARTICLE CONTENT:
{article_content}
"""

    prompt = f"""
You are analyzing recent startup funding news.

Your job has two parts:

1. Determine whether this article describes a startup or
technology company that recently raised or is actively
raising investment.

2. If it does, evaluate whether the company is a useful
job-search target for the following Product Manager.

CANDIDATE BACKGROUND:

- Around 8 years across product, engineering and consulting
- Product management
- Enterprise technology
- Data platforms
- AI products
- Generative AI
- Technical product management
- SaaS
- B2B
- Business transformation
- MBA
- Interested in growing technology companies

FIT SCORE:

9-10:
Excellent fit. Strong overlap with AI, enterprise software,
data, SaaS, fintech, healthcare technology, infrastructure,
workflow software, supply chain technology or B2B products.

7-8:
Strong potential company for Product Manager outreach.

5-6:
Possible opportunity but moderate relevance.

3-4:
Weak relevance.

1-2:
Poor target.

ACTION RULES:

8-10 = reach out now
5-7 = monitor
1-4 = skip

IMPORTANT:

- A venture capital firm raising a new VC fund is NOT a startup
  funding event.
- An investor announcing a fund is NOT a startup funding event.
- A weekly funding roundup describing many companies should
  normally be marked false because it is not one specific
  startup.
- An article only discussing an industry trend is NOT a
  specific startup funding event.
- An acquisition without a financing round is NOT automatically
  a funding event.
- Do not invent facts.
- If information is unavailable, use "Unknown".
- fit_score must be an integer from 1 through 10.
- Return ONLY valid JSON.
- Do not include markdown.
- Do not include explanation outside the JSON.

RETURN THIS EXACT STRUCTURE:

{{
  "is_funding_event": true,
  "company": "Company Name",
  "amount": "$25M",
  "stage": "Series A",
  "sector": "Enterprise AI",
  "key_people": "Founder or CEO",
  "hq": "San Francisco, US",
  "fit_score": 9,
  "action": "reach out now",
  "reason": "Short explanation of why this startup fits the candidate."
}}

ARTICLE:

{article_text}
"""

    try:

        response = (
            client
            .chat
            .completions
            .create(
                model=model_id,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            (
                                "You are a startup "
                                "funding data analyst. "
                                "Return valid JSON only."
                            ),
                    },
                    {
                        "role":
                            "user",
                        "content":
                            prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=600,
            )
        )

        raw_text = (
            response
            .choices[0]
            .message
            .content
        )

        result = parse_json_response(
            raw_text
        )

        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "Model output was JSON but "
                "not a JSON object."
            )

        # ---------------------------------------------------------
        # Defaults
        # ---------------------------------------------------------

        result.setdefault(
            "is_funding_event",
            False,
        )

        result.setdefault(
            "company",
            "Unknown",
        )

        result.setdefault(
            "amount",
            "Unknown",
        )

        result.setdefault(
            "stage",
            "Unknown",
        )

        result.setdefault(
            "sector",
            "Unknown",
        )

        result.setdefault(
            "key_people",
            "Unknown",
        )

        result.setdefault(
            "hq",
            "Unknown",
        )

        result.setdefault(
            "fit_score",
            5,
        )

        result.setdefault(
            "action",
            "monitor",
        )

        result.setdefault(
            "reason",
            "",
        )

        # ---------------------------------------------------------
        # Normalize funding flag
        # ---------------------------------------------------------

        funding_value = (
            result.get(
                "is_funding_event",
                False,
            )
        )

        if isinstance(
            funding_value,
            str,
        ):

            funding_value = (
                funding_value
                .strip()
                .lower()
                in [
                    "true",
                    "yes",
                    "1",
                ]
            )

        result[
            "is_funding_event"
        ] = bool(
            funding_value
        )

        # ---------------------------------------------------------
        # Normalize company
        # ---------------------------------------------------------

        company = str(
            result.get(
                "company",
                "Unknown",
            )
        ).strip()

        if not company:

            company = "Unknown"

        result[
            "company"
        ] = company

        # ---------------------------------------------------------
        # Normalize score
        # ---------------------------------------------------------

        try:

            score = int(
                result.get(
                    "fit_score",
                    5,
                )
            )

        except Exception:

            score = 5

        score = max(
            1,
            min(
                10,
                score,
            ),
        )

        result[
            "fit_score"
        ] = score

        # ---------------------------------------------------------
        # Force consistent action
        # ---------------------------------------------------------

        if score >= 8:

            result[
                "action"
            ] = "reach out now"

        elif score >= 5:

            result[
                "action"
            ] = "monitor"

        else:

            result[
                "action"
            ] = "skip"

        return result, None

    except Exception as e:

        return None, str(e)


# =============================================================================
# GOOGLE SHEETS
# =============================================================================

def export_to_sheets(df):

    try:

        if "GOOGLE_CREDS" not in st.secrets:

            st.warning(
                "GOOGLE_CREDS is missing "
                "from Streamlit secrets."
            )

            return False

        creds_value = (
            st.secrets[
                "GOOGLE_CREDS"
            ]
        )

        if isinstance(
            creds_value,
            str,
        ):

            creds_dict = json.loads(
                creds_value
            )

        else:

            creds_dict = dict(
                creds_value
            )

        scopes = [
            (
                "https://www.googleapis.com/"
                "auth/spreadsheets"
            ),
            (
                "https://www.googleapis.com/"
                "auth/drive"
            ),
        ]

        credentials = (
            Credentials
            .from_service_account_info(
                creds_dict,
                scopes=scopes,
            )
        )

        client = gspread.authorize(
            credentials
        )

        spreadsheet = client.open(
            "Startup Signal Tracker"
        )

        worksheet = (
            spreadsheet.sheet1
        )

        timestamp = (
            datetime
            .now(timezone.utc)
            .strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        )

        for _, row in df.iterrows():

            worksheet.append_row(
                [
                    timestamp,
                    row.get(
                        "company",
                        "",
                    ),
                    row.get(
                        "amount",
                        "",
                    ),
                    row.get(
                        "stage",
                        "",
                    ),
                    row.get(
                        "sector",
                        "",
                    ),
                    row.get(
                        "hq",
                        "",
                    ),
                    row.get(
                        "fit_score",
                        "",
                    ),
                    row.get(
                        "action",
                        "",
                    ),
                    row.get(
                        "reason",
                        "",
                    ),
                    row.get(
                        "key_people",
                        "",
                    ),
                    row.get(
                        "source",
                        "",
                    ),
                    row.get(
                        "link",
                        "",
                    ),
                ]
            )

        return True

    except Exception as e:

        st.error(
            f"Google Sheets export failed: {e}"
        )

        return False


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    model_id,
    use_jina=False,
):

    results = []

    success_count = 0
    failure_count = 0
    non_funding_count = 0
    unknown_company_count = 0

    error_messages = []

    try:

        client = get_groq_client()

    except Exception as e:

        st.error(
            f"Groq connection error: {e}"
        )

        return []

    entries = []

    # -------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------

    with st.status(
        "🔍 Scanning RSS feeds...",
        expanded=True,
    ) as status:

        entries = fetch_rss_entries()

        st.write(
            f"Found **{len(entries)}** "
            f"articles matching signals "
            f"in the last "
            f"{DAYS_WINDOW} days."
        )

        if not entries:

            status.update(
                label=(
                    "No matching articles found."
                ),
                state="complete",
            )

            return []

        st.write(
            f"Using Groq model: "
            f"**{model_id}**"
        )

        status.update(
            label=(
                "🤖 Analyzing funding signals..."
            )
        )

        progress_bar = st.progress(
            0
        )

        for index, entry in enumerate(
            entries
        ):

            article_content = ""

            if (
                use_jina
                and entry.get("link")
            ):

                article_content = (
                    fetch_article_content(
                        entry["link"]
                    )
                )

            analysis, error = (
                analyze_startup(
                    client=client,
                    model_id=model_id,
                    entry=entry,
                    article_content=
                        article_content,
                )
            )

            if error:

                failure_count += 1

                error_messages.append(
                    {
                        "title":
                            entry.get(
                                "title",
                                "",
                            ),
                        "source":
                            entry.get(
                                "source",
                                "",
                            ),
                        "error":
                            error,
                    }
                )

            elif not analysis.get(
                "is_funding_event",
                False,
            ):

                non_funding_count += 1

            else:

                company = str(
                    analysis.get(
                        "company",
                        "Unknown",
                    )
                ).strip()

                if (
                    not company
                    or company.lower()
                    == "unknown"
                ):

                    unknown_company_count += 1

                else:

                    success_count += 1

                    results.append(
                        {
                            "company":
                                company,

                            "amount":
                                analysis.get(
                                    "amount",
                                    "Unknown",
                                ),

                            "stage":
                                analysis.get(
                                    "stage",
                                    "Unknown",
                                ),

                            "sector":
                                analysis.get(
                                    "sector",
                                    "Unknown",
                                ),

                            "hq":
                                analysis.get(
                                    "hq",
                                    "Unknown",
                                ),

                            "key_people":
                                analysis.get(
                                    "key_people",
                                    "Unknown",
                                ),

                            "fit_score":
                                analysis.get(
                                    "fit_score",
                                    5,
                                ),

                            "action":
                                analysis.get(
                                    "action",
                                    "monitor",
                                ),

                            "reason":
                                analysis.get(
                                    "reason",
                                    "",
                                ),

                            "source":
                                entry.get(
                                    "source",
                                    "",
                                ),

                            "link":
                                entry.get(
                                    "link",
                                    "",
                                ),

                            "title":
                                entry.get(
                                    "title",
                                    "",
                                ),
                        }
                    )

            progress_bar.progress(
                (
                    index + 1
                )
                / len(entries)
            )

            # Small delay between model calls.
            time.sleep(
                0.4
            )

        # ---------------------------------------------------------
        # RESULTS SUMMARY
        # ---------------------------------------------------------

        st.write("---")

        st.write(
            "✅ Successfully analyzed: "
            f"**{success_count}**"
        )

        st.write(
            "⚠️ API / parsing failures: "
            f"**{failure_count}**"
        )

        st.write(
            "📰 Not actual funding events: "
            f"**{non_funding_count}**"
        )

        st.write(
            "❓ Company could not be identified: "
            f"**{unknown_company_count}**"
        )

        # Do NOT use st.expander here.
        # This is already inside st.status.

        if error_messages:

            st.warning(
                f"Groq encountered "
                f"{len(error_messages)} "
                f"error(s). Details below."
            )

            for error_item in error_messages:

                st.markdown(
                    "**Article:** "
                    + error_item[
                        "title"
                    ]
                )

                st.caption(
                    "Source: "
                    + error_item[
                        "source"
                    ]
                )

                st.code(
                    error_item[
                        "error"
                    ]
                )

                st.markdown(
                    "---"
                )

        if results:

            status.update(
                label=(
                    f"Done — "
                    f"{len(results)} "
                    f"startups scored"
                ),
                state="complete",
            )

        elif failure_count > 0:

            status.update(
                label=(
                    "Pipeline finished "
                    "with API errors."
                ),
                state="error",
            )

        else:

            status.update(
                label=(
                    "Pipeline finished — "
                    "no valid startup "
                    "funding events found."
                ),
                state="complete",
            )

    return sorted(
        results,
        key=lambda result:
            result.get(
                "fit_score",
                0,
            ),
        reverse=True,
    )


# =============================================================================
# RESULT CARD
# =============================================================================

def render_startup_card(
    result
):

    company = result.get(
        "company",
        "Unknown",
    )

    score = result.get(
        "fit_score",
        5,
    )

    action = result.get(
        "action",
        "monitor",
    )

    amount = result.get(
        "amount",
        "Unknown",
    )

    stage = result.get(
        "stage",
        "Unknown",
    )

    sector = result.get(
        "sector",
        "Unknown",
    )

    hq = result.get(
        "hq",
        "Unknown",
    )

    key_people = result.get(
        "key_people",
        "Unknown",
    )

    source = result.get(
        "source",
        "Unknown",
    )

    reason = result.get(
        "reason",
        "",
    )

    link = result.get(
        "link",
        "",
    )

    st.markdown(
        f"""
        <div class="startup-card">

            <div class="startup-title">
                {company}
            </div>

            <div class="startup-meta">
                <span class="score">
                    Fit: {score}/10
                </span>
                &nbsp; | &nbsp;
                {action.upper()}
            </div>

            <div class="startup-meta">
                💰 {amount}
                &nbsp; | &nbsp;
                📊 {stage}
                &nbsp; | &nbsp;
                🏭 {sector}
            </div>

            <div class="startup-meta">
                📍 {hq}
                &nbsp; | &nbsp;
                👤 {key_people}
            </div>

            <div class="startup-meta">
                📰 {source}
            </div>

            <p>
                {reason}
            </p>

            <a
                href="{link}"
                target="_blank"
            >
                Open article
            </a>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# LOAD AVAILABLE GROQ MODELS
# =============================================================================

available_models, model_error = (
    fetch_available_model_ids()
)

default_model = (
    choose_default_model(
        available_models
    )
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "## Abhishek Singh"
    )

    st.markdown(
        "Senior PM | AI Products | "
        "MBA UC Davis 2026"
    )

    st.markdown(
        "[LinkedIn]"
        "(https://www.linkedin.com/"
        "in/aabhishek-singh/) "
        "| "
        "[Portfolio]"
        "(https://abhishek-s-portfolio--"
        "abhisheksngh180.replit.app/)"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "## 🚀 Startup Signal Tracker"
    )

    st.markdown(
        "Monitors recent funding news "
        "and ranks startups by PM fit."
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### AI Model"
    )

    if model_error:

        st.error(
            "Could not retrieve Groq "
            f"models: {model_error}"
        )

        selected_model = None

    elif not available_models:

        st.error(
            "Groq returned no "
            "available models."
        )

        selected_model = None

    else:

        default_index = 0

        if (
            default_model
            in available_models
        ):

            default_index = (
                available_models
                .index(
                    default_model
                )
            )

        selected_model = (
            st.selectbox(
                "Groq model",
                available_models,
                index=
                    default_index,
                help=(
                    "This list comes "
                    "directly from your "
                    "Groq API key."
                ),
            )
        )

        st.success(
            f"Using: {selected_model}"
        )

    st.markdown(
        "---"
    )

    use_jina = st.toggle(
        "Deep article scan (Jina AI)",
        value=False,
        help=(
            "Fetch more article text "
            "before asking Groq to "
            "analyze it."
        ),
    )

    st.markdown(
        "---"
    )

    filter_action = (
        st.multiselect(
            "Action",
            [
                "reach out now",
                "monitor",
                "skip",
            ],
            default=[
                "reach out now",
                "monitor",
            ],
        )
    )

    filter_stage = (
        st.multiselect(
            "Stage",
            [
                "Pre-seed",
                "Seed",
                "Series A",
                "Series B",
                "Series C",
                "Series D",
                "Growth",
                "Unknown",
            ],
            default=[],
        )
    )

    filter_source = (
        st.multiselect(
            "Source",
            [
                name
                for name, _
                in RSS_FEEDS
            ],
            default=[],
        )
    )


# =============================================================================
# MAIN PAGE
# =============================================================================

st.title(
    "🚀 Startup Signal Tracker"
)

st.write(
    f"Live funding signals from "
    f"{len(RSS_FEEDS)} sources, "
    f"last {DAYS_WINDOW} days only."
)


# =============================================================================
# MODEL INFORMATION
# =============================================================================

if model_error:

    st.error(
        "Groq model discovery failed."
    )

    st.code(
        model_error
    )

elif available_models:

    st.caption(
        f"Groq API reports "
        f"{len(available_models)} "
        f"model(s) available to "
        f"this API key."
    )


# =============================================================================
# BUTTONS
# =============================================================================

col1, col2, col3 = (
    st.columns(
        [1, 1, 1]
    )
)


with col1:

    run_btn = st.button(
        "▶ Run Pipeline",
        type="primary",
        use_container_width=True,
    )


with col2:

    sheets_btn = st.button(
        "📊 Export to Sheets",
        use_container_width=True,
    )


with col3:

    models_btn = st.button(
        "🔍 Show Groq Models",
        use_container_width=True,
    )


# =============================================================================
# SHOW MODELS
# =============================================================================

if models_btn:

    if model_error:

        st.error(
            model_error
        )

    elif not available_models:

        st.warning(
            "Groq returned no models."
        )

    else:

        st.subheader(
            "Models Available "
            "to Your Groq API Key"
        )

        for model_id in (
            available_models
        ):

            st.code(
                model_id
            )


# =============================================================================
# RUN PIPELINE
# =============================================================================

if run_btn:

    if not selected_model:

        st.error(
            "No Groq model is "
            "available. Click "
            "'Show Groq Models' "
            "and inspect the result."
        )

    else:

        pipeline_results = (
            run_pipeline(
                model_id=
                    selected_model,
                use_jina=
                    use_jina,
            )
        )

        st.session_state[
            "results"
        ] = pipeline_results

        st.session_state[
            "run_time"
        ] = (
            datetime
            .now(timezone.utc)
            .strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        )

        st.session_state[
            "last_model"
        ] = selected_model


# =============================================================================
# SESSION RESULTS
# =============================================================================

results = (
    st.session_state.get(
        "results",
        [],
    )
)

run_time = (
    st.session_state.get(
        "run_time",
        "",
    )
)

last_model = (
    st.session_state.get(
        "last_model",
        "",
    )
)


# =============================================================================
# EXPORT
# =============================================================================

if sheets_btn:

    if not results:

        st.warning(
            "Run the pipeline before "
            "exporting."
        )

    else:

        dataframe = pd.DataFrame(
            results
        )

        if export_to_sheets(
            dataframe
        ):

            st.success(
                "Exported to "
                "Google Sheets."
            )


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if results:

    reach_count = sum(
        1
        for result in results
        if result.get(
            "action"
        ) == "reach out now"
    )

    monitor_count = sum(
        1
        for result in results
        if result.get(
            "action"
        ) == "monitor"
    )

    skip_count = sum(
        1
        for result in results
        if result.get(
            "action"
        ) == "skip"
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Total Startups",
        len(results),
    )

    metric2.metric(
        "🟢 Reach Out Now",
        reach_count,
    )

    metric3.metric(
        "🟡 Monitor",
        monitor_count,
    )

    metric4.metric(
        "🔴 Skip",
        skip_count,
    )

    if run_time:

        st.caption(
            f"Last run: {run_time}"
        )

    if last_model:

        st.caption(
            f"Model used: {last_model}"
        )

    # -------------------------------------------------------------
    # FILTERS
    # -------------------------------------------------------------

    filtered_results = (
        results.copy()
    )

    if filter_action:

        filtered_results = [
            result
            for result
            in filtered_results
            if result.get(
                "action"
            )
            in filter_action
        ]

    if filter_stage:

        filtered_results = [
            result
            for result
            in filtered_results
            if result.get(
                "stage"
            )
            in filter_stage
        ]

    if filter_source:

        filtered_results = [
            result
            for result
            in filtered_results
            if result.get(
                "source"
            )
            in filter_source
        ]

    st.subheader(
        f"Showing "
        f"{len(filtered_results)} "
        f"startups"
    )

    cards_tab, table_tab = (
        st.tabs(
            [
                "📋 Cards",
                "📊 Table",
            ]
        )
    )

    with cards_tab:

        if not filtered_results:

            st.info(
                "No startups match "
                "the current filters."
            )

        else:

            for result in (
                filtered_results
            ):

                render_startup_card(
                    result
                )

    with table_tab:

        if filtered_results:

            dataframe = pd.DataFrame(
                filtered_results
            )

            display_columns = [
                "company",
                "amount",
                "stage",
                "sector",
                "hq",
                "key_people",
                "fit_score",
                "action",
                "reason",
                "source",
                "link",
            ]

            display_columns = [
                column
                for column
                in display_columns
                if column
                in dataframe.columns
            ]

            st.dataframe(
                dataframe[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "link":
                        st.column_config
                        .LinkColumn(
                            "Article"
                        ),
                    "fit_score":
                        st.column_config
                        .NumberColumn(
                            "Fit Score",
                            min_value=1,
                            max_value=10,
                            format="%d/10",
                        ),
                },
            )

else:

    st.info(
        "Click **Run Pipeline** "
        "to scan for funding signals."
    )
