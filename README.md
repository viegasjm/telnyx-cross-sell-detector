# Telnyx Cross-Sell Signal Detector

Bot Week prototype — identifies cross-sell opportunities from account usage signals.

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Run with synthetic demo data
uvicorn app.main:app --host 0.0.0.0 --port 8420

# Or via Docker
docker-compose up
```

## API

See [API.md](API.md) for full endpoint reference (7 endpoints).

## Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Signal Detector | `services/signal_detector.py` | 5-category signal taxonomy |
| Inference Engine | `services/inference_engine.py` | Strict matching + confidence scoring |
| Account Scanner | `services/account_scanner.py` | Batch scan across accounts |
| Telnyx Client | `services/telnyx_client.py` | API integration (demo: anonymized data) |
| Cache | `services/cache.py` | Response caching layer |
| MCP Client | `services/mcp_client.py` | Model Context Protocol integration |

## Signal Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Usage** | "They're already doing the thing manually" | High SMS volume, no webhook |
| **Pain** | "They're hitting a wall" | High call failure rate |
| **Gap** | "They have X but not Y" | Voice without AI |
| **Fit** | "Profile matches proven adopters" | Healthcare vertical |
| **Suppression** | "Don't recommend" | Dormant account |

## Opportunity Tiers

- **Hot** (≥1.5): Route to account manager with personalized brief
- **Warm** (0.7–1.5): Show in portal dashboard
- **Cool** (<0.7): Log for batch analysis

## Demo Data

42 anonymized accounts in `data/demo/`. No PII — emails are `user_XXXX@demo.telnyx.com` format.

## Deployment

Designed for ACP agent consumption. The API returns structured JSON suitable for automated workflows. No assumptions about the caller.
