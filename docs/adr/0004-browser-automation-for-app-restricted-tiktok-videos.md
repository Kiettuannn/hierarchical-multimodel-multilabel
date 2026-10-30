# 0004 Browser Automation for App-Restricted TikTok Videos

## Status
Accepted

## Context
ADR 0003 established `yt-dlp` as the primary engine for TikTok channel video ingestion. However, TikTok imposes an "App-Restricted Video" policy on commercial and TikTok Shop videos ("Watch on app"), causing web endpoints to strip unwatermarked `playAddr` streams and leading to HTTP 403 Forbidden or audio-only downloads in `yt-dlp`. Direct headless HTTP scraping of third-party web resolver services (such as SaveTik) also fails due to Cloudflare Turnstile and anti-bot challenge blocks. For channels with high proportions of App-Restricted Videos, a browser-driven ingestion mechanism is required.

## Decision
We implement a dedicated browser automation ingestion script (`scripts/crawl_tiktok_playwright.py`) powered by Playwright Chromium:
1. **Channel Discovery via Browser**: Playwright navigates creator channel pages with human-like viewport and stealth parameters to scroll and harvest canonical video identifiers (`/video/<video_id>`).
2. **Browser Resolver for Streams**: For each un-archived video identifier, Playwright drives web resolver interfaces (e.g., SaveTik / SSSTik) to bypass Cloudflare challenges, locate the unwatermarked stream URL, and stream the media payload.
3. **Storage and Naming Contract**: Each downloaded file is saved strictly as `<video_id>.mp4` inside the designated output directory, maintaining full compatibility with the downstream multimodal feature extraction pipeline.
4. **Download Archive Idempotency**: Successful downloads append the identifier to the persistent `Download Archive` (`.download_archive.txt`), ensuring runs can be paused and resumed without duplicating network transfers.

## Considered Options
- **Updating yt-dlp Extractor Arguments**: Insufficient because TikTok completely removes web video stream descriptors for App-Restricted Videos, rendering desktop web extractors ineffective without device-signed mobile API tokens.
- **Paid Commercial Proxy/API (RapidAPI / TikAPI)**: Rejected for current usage to eliminate recurring external subscription overhead, though retained as a potential upgrade if high-concurrency ingestion becomes necessary.
- **Direct HTTP Scraper against SaveTik**: Rejected because SaveTik blocks direct script requests with connection timeouts and Cloudflare challenges.

## Consequences
- Adds `playwright` and the Chromium browser binary to ingestion environment dependencies.
- Increases execution latency per item (approx. 3–7 seconds per video) compared to direct HTTP streaming, which remains acceptable under the low-frequency ingestion requirement.
- Guarantees retrieval of unwatermarked Raw Video files for App-Restricted Videos that fail under `yt-dlp`.
