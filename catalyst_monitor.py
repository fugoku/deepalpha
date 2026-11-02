"""Public-headline catalyst monitoring for Canadian crypto paper-trading context."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
CATALYST_QUERY = (
    'bitcoin OR crypto OR cryptocurrency OR "digital assets" OR '
    '"Bank of Canada" OR "Federal Reserve" OR Powell OR "interest rates" OR '
    'inflation OR tariff OR sanctions OR "King Charles" OR president OR '
    '"prime minister" OR election OR war OR hack OR stablecoin OR ETF OR '
    'exchange OR regulation'
)
LEADER_TERMS = (
    "president", "prime minister", "king charles", "royal address",
    "federal reserve chair", "fed chair", "powell", "bank of canada governor",
    "treasury secretary", "finance minister", "sec chair",
)
SPEECH_TERMS = (
    "speaks", "speaking", "speech", "remarks", "address", "testimony",
    "press conference", "announces", "announced", "statement",
)
HIGH_IMPACT_TERMS = (
    "rate decision", "interest rate", "emergency meeting", "sanctions",
    "tariff", "military strike", "invasion", "ceasefire", "war",
    "exchange hack", "crypto hack", "security breach", "bank failure",
    "stablecoin depeg", "etf approval", "etf rejection", "ban on crypto",
    "crypto ban", "bitcoin reserve",
)
NEWS_TERMS = (
    "crypto", "bitcoin", "ethereum", "digital asset", "blockchain",
    "exchange", "stablecoin", "interest rate", "inflation", "tariff",
    "sanction", "central bank", "federal reserve", "bank of canada", "market",
)


def _published_at(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def classify_headline(
    title: str,
    tone: float | None = None,
) -> dict[str, Any] | None:
    """Keep relevant headlines and estimate risk context, not trading direction."""
    normalized = title.casefold()
    mentions_leader = any(term in normalized for term in LEADER_TERMS)
    if not any(term in normalized for term in NEWS_TERMS) and not mentions_leader:
        return None

    mentions_speech = any(term in normalized for term in SPEECH_TERMS)
    high_impact = any(term in normalized for term in HIGH_IMPACT_TERMS)
    if mentions_leader and mentions_speech:
        category = "leader_speech"
        high_impact = True
    elif any(term in normalized for term in ("hack", "breach", "exploit")):
        category = "security"
        high_impact = True
    elif any(term in normalized for term in ("rate", "inflation", "central bank", "powell")):
        category = "monetary_policy"
    elif any(term in normalized for term in ("regulation", "sec ", "etf", "ban on crypto")):
        category = "regulatory"
    elif any(term in normalized for term in ("war", "strike", "sanction", "tariff", "ceasefire")):
        category = "geopolitical"
        high_impact = True
    else:
        category = "market_news"

    try:
        tone_value = float(tone) if tone is not None else 0.0
    except (TypeError, ValueError):
        tone_value = 0.0
    return {
        "category": category,
        "impact": "high" if high_impact else "watch",
        "tone": tone_value,
        "mentions_leader": mentions_leader,
        "mentions_speech": mentions_speech,
        "risk_note": (
            "High-impact catalyst; pause new paper entries for review."
            if high_impact else "Headline context only; not a directional signal."
        ),
    }


def fetch_market_catalysts(
    *,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Retrieve recent public GDELT headlines relevant to crypto and market catalysts."""
    client = session or requests.Session()
    response = client.get(
        GDELT_DOC_API,
        params={
            "query": CATALYST_QUERY,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 100,
            "timespan": "24h",
            "sort": "datedesc",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("articles", []), list):
        raise ValueError("GDELT returned an unexpected article response.")

    current_time = now or datetime.now(timezone.utc)
    events = []
    seen_urls = set()
    for article in payload.get("articles", []):
        title = str(article.get("title") or "").strip()
        url = str(article.get("url") or "").strip()
        published = _published_at(article.get("seendate", ""))
        if not title or not url or url in seen_urls or published is None:
            continue
        if published > current_time + timedelta(minutes=5):
            continue
        classification = classify_headline(title, article.get("tone"))
        if classification is None:
            continue
        seen_urls.add(url)
        events.append({
            "title": title,
            "url": url,
            "source": str(article.get("domain") or ""),
            "published_at": published.isoformat(),
            "source_country": str(article.get("sourcecountry") or ""),
            **classification,
        })
    return events


def recent_high_impact_events(
    events: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    pause_minutes: int = 60,
) -> list[dict[str, Any]]:
    """Select recent high-impact headlines for conservative paper-entry throttling."""
    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(minutes=pause_minutes)
    recent = []
    for event in events:
        published = datetime.fromisoformat(event["published_at"])
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if event.get("impact") == "high" and cutoff <= published <= current_time:
            recent.append(event)
    return sorted(recent, key=lambda event: event["published_at"], reverse=True)
