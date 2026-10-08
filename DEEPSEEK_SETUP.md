# V2.3 — optional real DeepSeek knowledge Q&A

Status: integration code prepared; real service must be deployed and tested before use. No provider key is present in this repository, archive, HTML or JavaScript.

## Architecture

GitHub Pages frontend -> authenticated gateway -> official DeepSeek API.

The frontend remains on the existing gh-pages branch. The gateway and Render blueprint live on piston-ai-demo. The research main branch is not changed. GitHub Pages cannot execute this gateway.

## Backend deployment

Use the root render.yaml blueprint from branch piston-ai-demo, or create a Render Web Service:
- Root directory: server
- Runtime: Node 22+
- Build: node --check server.mjs
- Start: node server.mjs
- Health path: /healthz
- Use the free service plan for the demo; this does not make DeepSeek inference free.

Environment variables (set privately in the deployment dashboard, NEVER in Git):
- DEEPSEEK_API_KEY: your active DeepSeek key
- DEMO_ACCESS_TOKEN: a separate random 32+ character passphrase. Blueprint generates it. Never use the provider key as this passphrase.
- DEEPSEEK_MODEL: deepseek-flash (current official name checked 2026-10-08)
- ALLOWED_ORIGINS: https://ustbtobyma.github.io (origin only, no repository path)
- MAX_OUTPUT_TOKENS: 1800
- REQUESTS_PER_HOUR: 60

In the knowledge page, choose “连接 DeepSeek”. Enter the deployed gateway URL and DEMO_ACCESS_TOKEN, accept data transfer, and test. This performs one tiny real model completion, not merely a health check. Only successful tests unlock online mode. Refreshing clears the access passphrase. The provider key must never be typed into this browser dialog.

## Local backend

Copy server/.env.example to server/.env locally, fill the two secrets there, and run:

    cd server
    node --env-file=.env server.mjs

Serve the site directory using a local static server on port 8765. Set ALLOWED_ORIGINS to include http://127.0.0.1:8765, then set the gateway URL to http://127.0.0.1:3000. Local files opened via file:// are intentionally not an allowed origin for real-mode use.

## Safety and limitations

Only selected evidence is sent; TXT/MD/CSV file use requires separate consent. Selected files are capped to a documented initial excerpt and recent history is limited. Text is not represented as a vector search result. No PDF parsing, image recognition or web browsing is added. Files remain absent from GitHub. Live mode never silently replaces failed requests with demo answers.

The gateway has an independent passphrase, allowlisted origins, fixed upstream host, request and output bounds, no automatic retries, two concurrent model calls maximum, and 60 requests/hour by default. Counters are single-process, reset on restart, and are NOT a durable financial cap. Public production requires stronger user authentication, durable quotas and monitoring. Do not publish the access passphrase. Model output and user content are escaped and no model-produced HTML or images are executed.

Provider credentials already pasted into a chat should be rotated before public business demonstrations; enter the replacement only into the backend's secret configuration, not into chat or GitHub.

## Verification

Run node --test server/server.test.mjs. Automated tests use a local mock upstream and do not validate an actual DeepSeek credential. Real endpoint access and account balance must still be checked in the deployed environment.

Official references:
- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/api/create-chat-completion/
- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages
- https://render.com/docs/configure-environment-variables
- https://render.com/docs/blueprint-spec
