# 0003 TikTok Channel Crawler Engine and Ingestion Strategy

## Status
Accepted

## Context
The dataset pipeline requires acquiring unwatermarked short-form raw videos from TikTok creator channels to expand the multimodal harmfulness dataset. TikTok frequently modifies web APIs, obfuscates query signatures (msToken, XBogus), and implements strict anti-bot rate limits and IP throttling. The ingestion tool must retrieve videos reliably, store files named strictly by their canonical platform video identifier (`<video_id>.mp4`), avoid watermarked frames, and support idempotent resuming across large creator catalogues.

## Decision
We adopt `yt-dlp` as the core crawler and extraction engine wrapped in a dedicated CLI utility (`scripts/crawl_tiktok_channel.py`):
1. **Engine Selection**: Use `yt-dlp`'s native TikTok extractor to resolve channel feeds, navigate pagination, and extract direct unwatermarked CDN video streams without launching headless browser instances.
2. **File and Naming Contract**: Every video is saved strictly as `<video_id>.mp4` inside a user-defined `--output-dir`. Sidecar metadata files are omitted by default to keep the raw video workspace clean, with an optional `--save-metadata` flag providing an exported CSV manifest when needed.
3. **Idempotency via Download Archive**: Utilize persistent download archives (`--archive-file`) to record completed video IDs, preventing duplicate downloads and allowing safe interruption and resumption.
4. **Polite Pacing**: Enforce randomized sleep delays between extraction requests and implement exponential backoff on HTTP 429 responses to mitigate IP blocks.

## Considered Options
- **Headless Browser Automation (Playwright/Selenium)**: Rejected due to heavy resource overhead (requiring browser binaries, high RAM/CPU), slow DOM-scrolling speeds, and fragility against dynamic DOM selector updates.
- **Direct Reverse-Engineered TikTok Web API**: Rejected due to high maintenance burden and frequent breakages caused by signature rotation and encryption parameter updates.

## Consequences
- Requires `yt-dlp` in the project dependencies (or virtual environment).
- Fast, headless ingestion capable of running in CLI and remote environments.
- Video filenames match the standard `<video_id>.mp4` convention used throughout the downstream video processing and metadata extraction pipelines.
