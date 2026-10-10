# V2.2: answer-first knowledge demonstration

This update adds an evidence-based explanation layer to the existing V2.1 large-font interface. It is a deterministic demo, not an LLM, and does not make network requests to model services.

## Changes

- Lead with a direct answer, then explain calculations, limits and validation steps.
- Collapse raw data into an expandable evidence section.
- Support follow-up questions and retain up to 16 turns for the current session.
- Allow explicitly selected TXT / MD / CSV files to participate in this session's answers.
- CSV: choose a numerical column, calculate differences, show source rows, and retain material / temperature filters for follow-ups.
- Text: extract matching paragraphs, explain evidence limits, show original line numbers. This is not general semantic understanding.
- Preserve the large-font controls, dual-brand header, process optimization and main research branch.

## Limits and privacy

No general LLM, PDF parsing, image recognition, vector database, causal inference or production-system integration. Uploaded text is escaped before display and never treated as code or instructions. Added files are not sent to GitHub or a backend. Session files and conversation history are cleared on reload.

Text analysis is capped at 80,000 characters. CSV analysis is capped at 1,000,000 characters / 5,000 records and 40 columns. Unsupported formats and absent evidence must not be represented as model predictions.

## Smoke tests

Locally exercised in headless Chromium: default answer, causal-explanation follow-up, built-in sample CSV, new CSV values changing the calculated answer, 350°C filtering retained on follow-up, missing-temperature response, text rate-difference explanation, escaped HTML input, existing process optimization and 390px mobile layout. No page JavaScript errors observed in those checks. This is not a claim of exhaustive online testing.

## Future real-model integration

Keep model credentials on an authenticated backend. Do not expose API keys in browser code or commit them into a public repository. Explain external data transfer before enabling a model service for company documents.
