# Cross-Sell Signal Detector — API Reference

Base URL: `http://localhost:8420`

## Endpoints

### `GET /api/accounts`
List all accounts with signal data.

**Response:**
```json
{
  "total": 42,
  "accounts": [
    {
      "user_id": "uuid",
      "business_name": "string",
      "email": "user_XXXX@demo.telnyx.com",
      "country": "US",
      "has_voice": true,
      "has_messaging": false,
      "has_ai": false,
      "total_connections": 3,
      "total_numbers": 25,
      "opportunity_score": 3.83,
      "signal_count": 6,
      "signals": [...],
      "recommendations": [...]
    }
  ]
}
```

### `GET /api/account/{user_id}`
Full detail for a single account.

**Response:** Account object with full `signals` and `recommendations` arrays.

### `GET /api/signals`
Signal distribution across all accounts.

**Response:**
```json
{
  "distribution": {
    "signal_name": { "count": N, "avg_confidence": 0.XX }
  }
}
```

### `GET /api/recommendations`
Recommendation summary across all accounts.

**Response:**
```json
{
  "summary": {
    "target_product": { "account_count": N, "avg_confidence": 0.XX, "est_revenue_monthly": N }
  }
}
```

### `GET /api/refresh`
Reload data from disk (re-reads JSON files).

## Data Model

### Signal
| Field | Type | Description |
|-------|------|-------------|
| name | string | Signal identifier (e.g. `no_ai_assistant`, `high_inbound_call_volume`) |
| category | string | SignalCategory enum: `voice_usage`, `messaging_usage`, `tech_stack`, `pain_point`, `compliance`, `quality`, `negative` |
| strength | string | SignalStrength: `strong`, `moderate`, `weak` |
| confidence | float | 0.0–1.0 confidence in this signal |
| description | string | Human-readable explanation |
| evidence | list[str] | Supporting evidence bullets |

### Recommendation
| Field | Type | Description |
|-------|------|-------------|
| title | string | Recommendation title |
| target_product | string | Telnyx product to cross-sell |
| target_feature | string | Specific feature within product |
| priority | int | 1=highest |
| confidence | float | 0.0–1.0 |
| integration_effort | string | `Low`, `Medium`, `High` |
| reasoning_chain | list[str] | Step-by-step reasoning |
| evidence_bullets | list[str] | Supporting evidence |
| estimated_value | object | `{customer_savings_monthly, telnyx_revenue_monthly, retention_uplift}` |
| integration_path | list[str] | Step-by-step integration guide |

### Opportunity Score
Sum of (signal.confidence × signal.strength_weight) across all positive signals.
- **Hot**: ≥ 1.5
- **Warm**: ≥ 0.7
- **Cool**: < 0.7

## Bot Week Demo

Demo data is in `data/demo/` (anonymized). Raw data stays in `data/`. The app prefers `data/demo/` when present.

## Deployment (Docker)

```bash
docker-compose up -d
```

Environment variable: `TELNYX_API_KEY` — required only for live crawling (not needed for demo mode with pre-generated data).
