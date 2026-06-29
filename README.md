# TechOps Docsite Update Automation

FastAPI prototype for a GitHub App that detects merged source PRs and opens AI-generated documentation PRs against a separate MkDocs repo.

## AI Provider

The default provider is deterministic and local:

```bash
AI_PROVIDER=mock
```

To use OpenRouter with DeepSeek:

```bash
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=deepseek/deepseek-v3.2
```

Optional OpenRouter settings:

```bash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://github.com/bnb-chain/techops-docsite-update-automation
OPENROUTER_APP_TITLE="AI Docsite Update Prototype"
```

The OpenRouter client requests strict JSON output and rejects invalid responses instead of creating documentation PRs.

## Test

```bash
python3 -m pytest
```

## Demo

```bash
PYTHONPATH=src python3 scripts/e2e_demo.py
```

## Real GitHub App Path

Set these environment variables to use the real GitHub App client and OpenRouter:

```bash
GITHUB_CLIENT=app
GITHUB_APP_ID=your_github_app_id
GITHUB_PRIVATE_KEY='-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----'
GITHUB_WEBHOOK_SECRET=your_webhook_secret
DOCSITE_REPO=koredeBNB/mock-mkdocs-repo

AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=deepseek/deepseek-v3.2
```

Run the webhook server:

```bash
set -a
source .env
set +a

PYTHONPATH=src uvicorn docsite_updater.main:app --reload --port 8000
```

Expose it with a tunnel:

```bash
ngrok http 8000
```

Use the tunnel URL as the GitHub App webhook URL:

```text
https://your-tunnel-url/webhooks/github
```

Install the GitHub App on:

- `koredeBNB/mock-bsc-app`
- `koredeBNB/mock-mkdocs-repo`

Required repository permissions:

- Metadata: read-only
- Contents: read/write
- Pull requests: read/write

Required event:

- Pull request
