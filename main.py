import requests
from bs4 import BeautifulSoup
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import html
from typing import Dict, List, Optional
import os


# ==========================================
# CONFIG
# ==========================================
# Tạm thời hard-code theo yêu cầu.
# Khuyến nghị: sau khi chạy ổn, chuyển sang biến môi trường.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

SPREADSHEET_NAME = "MEXC Sitemap Checker"
WORKSHEET_NAME = "List"

MODEL = "gpt-5-nano"

# Số luồng quét sitemap.
MAX_WORKERS_SITEMAP = 5

# Không nên để quá cao vì dễ bị rate limit / timeout / tốn tiền nhanh.
MAX_WORKERS_AI = 50

# Lọc ngày theo GMT+7.
TZ_GMT7 = timezone(timedelta(hours=7))

# Nếu muốn lấy hôm qua: 1
# Nếu muốn lấy hôm kia: 2
DAYS_BACK = 1

INPUT_SITEMAPS = [
    "https://www.mexc.co/uk-UA/news/all-sitemap.xml",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SHEET_HEADER = [
    "URL",
    "Summary",
    "Input Tokens",
    "Output Tokens",
    "Total Tokens",
    "Sentiment",
    "Topic",
    "Reason",
    "Keep/Remove",
    "Language",
    "Date",
    "Title",
    "Content",
    "Run Date",
]


# ==========================================
# TEXT / HTML HELPERS
# ==========================================

def clean_html_to_text(html_content: str) -> str:
    """Convert HTML to clean plain text."""
    if not html_content:
        return ""

    text = html.unescape(str(html_content))
    text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split()).strip()


def extract_content_mexc(html_text: str) -> Dict[str, str]:
    """Extract title and content from MEXC news article HTML."""
    soup = BeautifulSoup(html_text, "html.parser")

    h1 = soup.find("h1", class_="detail-header_title__FLt9q")
    title_text = h1.get_text(" ", strip=True) if h1 else ""

    content_div = soup.find("div", id="news-rich-content")
    content_text = clean_html_to_text(str(content_div)) if content_div else ""

    return {
        "title": title_text,
        "content": content_text,
    }


def clip_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " [...]"


def normalize_topic(topic_raw: str) -> str:
    t = (topic_raw or "").lower()

    if "bitcoin" in t:
        return "Bitcoin"
    if "ethereum" in t:
        return "Ethereum"
    if "solana" in t:
        return "Solana"
    if "altcoin" in t or "token" in t:
        return "Altcoin"
    if "not related" in t or "unrelated" in t:
        return "not related to crypto"

    return "crypto headlines"


def normalize_sentiment(sentiment_raw: str) -> str:
    t = (sentiment_raw or "").lower().strip()

    if t == "positive":
        return "positive"
    if t == "negative":
        return "negative"

    return "neutral"


def normalize_ai_output(raw_output: str) -> Dict[str, str]:
    """
    Normalize AI output into:
    sentiment; topic; reason; decision

    Keep by default.
    Remove only if the model explicitly returns Remove.
    """
    s = (raw_output or "").strip()
    s = re.sub(r"```[\s\S]*?```", "", s).replace("\r", "")

    first_line = ""
    for line in s.split("\n"):
        line = line.strip()
        if line:
            first_line = line
            break

    if not first_line:
        first_line = s

    # Fallback: support pipe separator if model accidentally uses it.
    if ";" not in first_line and "|" in first_line:
        first_line = first_line.replace("|", ";")

    parts = [p.strip() for p in first_line.split(";")]

    sentiment = normalize_sentiment(parts[0] if len(parts) > 0 else "neutral")
    topic = normalize_topic(parts[1] if len(parts) > 1 else "")
    reason = parts[2] if len(parts) > 2 and parts[2] else "Insufficient signal"

    decision_raw = parts[3].strip().lower() if len(parts) > 3 else ""

    # IMPORTANT:
    # Keep by default. Remove only if explicitly "Remove".
    decision = "Remove" if decision_raw == "remove" else "Keep"

    # Consistency guard.
    reason_l = reason.lower()
    if decision == "Remove" and (
        "no policy violation" in reason_l
        or "no violation" in reason_l
    ):
        decision = "Keep"

    summary = f"{sentiment}; {topic}; {reason}; {decision}"

    return {
        "Summary": summary,
        "Sentiment": sentiment,
        "Topic": topic,
        "Reason": reason,
        "Decision": decision,
    }


# ==========================================
# OPENAI
# ==========================================

def build_gatekeeper_prompt(title: str, content: str) -> str:
    """Build strict classification prompt based on the original Apps Script rules."""
    return f"""
You are a strict content quality gatekeeper for MEXC News.

Return EXACTLY ONE LINE with 4 fields separated by semicolons:
<sentiment>; <topic>; <reason>; <decision>

Rules:
- One line only
- No quotes, no markdown
- Do not use semicolons inside reason

Sentiment (about MEXC only):
positive | negative | neutral
If MEXC is not mentioned or discussed → neutral

Topic:
Bitcoin | Ethereum | Solana | Altcoin | crypto headlines | not related to crypto

Decision:
Keep | Remove

Decision rules (IMPORTANT):
- Keep by default.
- Remove ONLY if the content includes or promotes:
  - adult or sexual content
  - casino, betting, gambling, or lottery
  - scams, fraud, phishing, or financial deception aimed at readers
  - defamation or unverified criminal accusations
  - content that clearly harms the MEXC brand or reputation

Important distinction:
- Reporting on scams, fraud, phishing, hacks, lawsuits, or criminal accusations as normal news is allowed.
- Remove only when the article itself promotes, enables, or makes unverified harmful accusations.

Reason for Keep or Remove:
- The reason must justify the DECISION only
- Never use "No mention of MEXC" as a reason by itself
- If decision = Keep: write "No policy violation" + one concrete safe signal, such as market news, exchange update, technical analysis, regulation news, or project news
- If decision = Remove: name the specific violation, such as promotes gambling, adult content, phishing link, or unverified criminal accusation

Title:
{clip_text(title, 500)}

Content:
{clip_text(content, 12000)}
""".strip()


def call_openai_gatekeeper(title: str, content: str) -> Dict[str, object]:
    """Call OpenAI and return normalized classification result."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        return {
            "Summary": "error; crypto headlines; Missing OpenAI API key; Remove",
            "Input Tokens": 0,
            "Output Tokens": 0,
            "Total Tokens": 0,
            "Sentiment": "neutral",
            "Topic": "crypto headlines",
            "Reason": "Missing OpenAI API key",
            "Decision": "Remove",
        }

    prompt = build_gatekeeper_prompt(title, content)

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()

        data = resp.json()

        raw_content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        usage = data.get("usage", {}) or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)

        normalized = normalize_ai_output(raw_content)

        return {
            **normalized,
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Total Tokens": total_tokens,
        }

    except Exception as e:
        return {
            "Summary": "error; crypto headlines; OpenAI call failed; Remove",
            "Input Tokens": 0,
            "Output Tokens": 0,
            "Total Tokens": 0,
            "Sentiment": "neutral",
            "Topic": "crypto headlines",
            "Reason": f"OpenAI call failed: {e}",
            "Decision": "Remove",
        }


# ==========================================
# SITEMAP
# ==========================================

def process_single_sitemap(sitemap_url: str, target_date_str: str) -> List[Dict[str, str]]:
    """Process one news sitemap and return articles matching target date."""
    results = []

    try:
        response = requests.get(
            sitemap_url,
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        urls = soup.find_all("url")

        for u in urls:
            news_tag = u.find("news:news")
            if not news_tag:
                continue

            pub_date_tag = news_tag.find("news:publication_date")
            pub_date_raw = pub_date_tag.get_text(strip=True) if pub_date_tag else ""

            if pub_date_raw[:10] != target_date_str:
                continue

            loc_tag = u.find("loc")
            title_tag = news_tag.find("news:title")

            pub_info = news_tag.find("news:publication")
            pub_name_tag = pub_info.find("news:name") if pub_info else None
            pub_lang_tag = pub_info.find("news:language") if pub_info else None

            results.append({
                "URL": loc_tag.get_text(strip=True) if loc_tag else "",
                "Publication Name": pub_name_tag.get_text(strip=True) if pub_name_tag else "",
                "Language": pub_lang_tag.get_text(strip=True) if pub_lang_tag else "",
                "Date": pub_date_raw,
                "Title": title_tag.get_text(" ", strip=True) if title_tag else "",
            })

        return results

    except Exception as e:
        print(f"[SITEMAP ERROR] {sitemap_url}: {e}")
        return []


def get_news_from_sitemaps(
    sitemap_list: List[str],
    target_date_str: str,
    max_workers: int = 5,
) -> List[Dict[str, str]]:
    """Fetch all matching news URLs from all sitemaps."""
    all_results = []

    print(f"--- Filtering news date: {target_date_str}, sitemap workers: {max_workers} ---")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(process_single_sitemap, url, target_date_str): url
            for url in sitemap_list
        }

        for future in as_completed(future_to_url):
            sitemap_url = future_to_url[future]

            try:
                data = future.result()
                print(f"-> {sitemap_url}: found {len(data)} matching articles.")
                all_results.extend(data)

            except Exception as e:
                print(f"-> {sitemap_url}: failed: {e}")

    # De-duplicate URLs from sitemap results.
    seen = set()
    deduped = []

    for item in all_results:
        url = (item.get("URL") or "").strip()
        if not url or url in seen:
            continue

        seen.add(url)
        deduped.append(item)

    return deduped


# ==========================================
# ARTICLE PROCESSING
# ==========================================

def fetch_article_html(url: str) -> str:
    resp = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()

    text = resp.text or ""
    if len(text) < 200:
        raise ValueError("HTML too short")

    return text


def make_error_result(
    url_data: Dict[str, str],
    reason: str,
    run_date: str,
) -> Dict[str, object]:
    """Return a row-shaped error result."""
    return {
        "URL": url_data.get("URL", ""),
        "Summary": "error; crypto headlines; Content error or fetch failed; Remove",
        "Input Tokens": 0,
        "Output Tokens": 0,
        "Total Tokens": 0,
        "Sentiment": "neutral",
        "Topic": "crypto headlines",
        "Reason": reason,
        "Decision": "Remove",
        "Language": url_data.get("Language", ""),
        "Date": url_data.get("Date", ""),
        "Title": url_data.get("Title", ""),
        "Content": "",
        "Run Date": run_date,
    }


def process_url_full_logic(url_data: Dict[str, str], run_date: str) -> Optional[Dict[str, object]]:
    """Fetch article, extract content, classify, and return final row."""
    url = (url_data.get("URL") or "").strip()

    if not url:
        return None

    try:
        html_text = fetch_article_html(url)
        extracted = extract_content_mexc(html_text)

        scraped_title = extracted.get("title") or ""
        content_text = extracted.get("content") or ""

        final_title = scraped_title or url_data.get("Title", "")

        if len(content_text) < 30:
            ai_res = {
                "Summary": "error; crypto headlines; Content too short; Remove",
                "Input Tokens": 0,
                "Output Tokens": 0,
                "Total Tokens": 0,
                "Sentiment": "neutral",
                "Topic": "crypto headlines",
                "Reason": "Content too short or cannot extract article body",
                "Decision": "Remove",
            }
        else:
            ai_res = call_openai_gatekeeper(final_title, content_text)

        return {
            "URL": url,
            "Summary": ai_res.get("Summary", ""),
            "Input Tokens": ai_res.get("Input Tokens", 0),
            "Output Tokens": ai_res.get("Output Tokens", 0),
            "Total Tokens": ai_res.get("Total Tokens", 0),
            "Sentiment": ai_res.get("Sentiment", "neutral"),
            "Topic": ai_res.get("Topic", "crypto headlines"),
            "Reason": ai_res.get("Reason", ""),
            "Decision": ai_res.get("Decision", "Remove"),
            "Language": url_data.get("Language", ""),
            "Date": url_data.get("Date", ""),
            "Title": final_title,
            "Content": content_text,
            "Run Date": run_date,
        }

    except Exception as e:
        return make_error_result(
            url_data=url_data,
            reason=f"Content fetch/extract failed: {e}",
            run_date=run_date,
        )


def process_articles_parallel(
    raw_list: List[Dict[str, str]],
    run_date: str,
    max_workers: int = 5,
) -> List[Dict[str, object]]:
    """Process article URLs in parallel."""
    final_data = []

    print(f"--- Processing {len(raw_list)} articles, AI workers: {max_workers} ---")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_url_full_logic, item, run_date)
            for item in raw_list
        ]

        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                res = future.result()

                if res:
                    final_data.append(res)
                    print(
                        f"[{idx}/{len(futures)}] {res.get('Decision')} | "
                        f"{res.get('Topic')} | {res.get('URL')}"
                    )

            except Exception as e:
                print(f"[ARTICLE ERROR] {e}")

    return final_data


# ==========================================
# GOOGLE SHEETS
# ==========================================

def get_worksheet():
    """Authorize and return target worksheet."""
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json",
        scopes,
    )

    client = gspread.authorize(creds)
    spreadsheet = client.open(SPREADSHEET_NAME)

    try:
        return spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        return spreadsheet.sheet1


def ensure_header(sheet) -> None:
    """Create or fix header row if sheet is empty."""
    values = sheet.get_all_values()

    if not values:
        sheet.append_row(SHEET_HEADER)
        return

    first_row = values[0]

    # If A1 is empty, update first row.
    if not first_row or not first_row[0].strip():
        sheet.update("A1:N1", [SHEET_HEADER])
        return

    # If existing sheet has a different structure, do not overwrite automatically.
    # This avoids destroying your old Apps Script sheet format.
    # New rows are still appended in the Python structure.


def get_existing_urls(sheet) -> set:
    """Read existing URL column to prevent duplicates."""
    try:
        urls = sheet.col_values(1)
    except Exception:
        return set()

    return {
        u.strip()
        for u in urls[1:]
        if u and u.strip()
    }


def append_results_to_sheet(final_data: List[Dict[str, object]]) -> int:
    """Append final processed data to Google Sheet, skipping duplicate URLs."""
    if not final_data:
        return 0

    sheet = get_worksheet()
    ensure_header(sheet)

    # existing_urls = get_existing_urls(sheet)

    rows_to_append = []

    for r in final_data:
        url = (r.get("URL") or "").strip()

        # if not url or url in existing_urls:
        #     continue

        if not url:
            continue

        rows_to_append.append([
            r.get("URL", ""),
            r.get("Summary", ""),
            r.get("Input Tokens", 0),
            r.get("Output Tokens", 0),
            r.get("Total Tokens", 0),
            r.get("Sentiment", ""),
            r.get("Topic", ""),
            r.get("Reason", ""),
            r.get("Decision", ""),
            r.get("Language", ""),
            r.get("Date", ""),
            r.get("Title", ""),
            r.get("Content", ""),
            r.get("Run Date", ""),
        ])

        # existing_urls.add(url)

    if rows_to_append:
        sheet.append_rows(
            rows_to_append,
            value_input_option="USER_ENTERED",
        )

    return len(rows_to_append)


# ==========================================
# MAIN
# ==========================================

def main():
    start_time = time.time()

    target_date = (datetime.now(TZ_GMT7) - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    run_date = datetime.now(TZ_GMT7).strftime("%Y-%m-%d")

    print("=" * 80)
    print("MEXC News Sitemap Checker")
    print(f"Target date: {target_date}")
    print(f"Run date: {run_date}")
    print("=" * 80)

    raw_list = get_news_from_sitemaps(
        sitemap_list=INPUT_SITEMAPS,
        target_date_str=target_date,
        max_workers=MAX_WORKERS_SITEMAP,
    )

    if not raw_list:
        print(f"No articles found for {target_date}.")
        return

    print(f"Total unique URLs from sitemap: {len(raw_list)}")

    final_processed_data = process_articles_parallel(
        raw_list=raw_list,
        run_date=run_date,
        max_workers=MAX_WORKERS_AI,
    )

    if not final_processed_data:
        print("No processed data.")
        return

    appended_count = append_results_to_sheet(final_processed_data)

    elapsed = time.time() - start_time

    print("=" * 80)
    print(f"Done. Processed: {len(final_processed_data)} articles.")
    print(f"Appended to sheet: {appended_count} new rows.")
    # print(f"Skipped duplicates: {len(final_processed_data) - appended_count}.")
    print(f"Appended rows: {appended_count}.")
    print(f"Elapsed: {elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()