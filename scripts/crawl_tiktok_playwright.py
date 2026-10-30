"""TikTok Channel Video Crawler using Playwright Browser Automation with Stealth.

This crawler retrieves unwatermarked raw video files (<video_id>.mp4) from any
TikTok creator channel, including App-Restricted Videos (TikTok Shop / "Watch on app")
that fail under standard tools with HTTP 403 Forbidden or audio-only streams.

Architecture:
- Phase 1 (Catalogue Discovery): Uses an isolated stealth browser to navigate the
  creator profile, harvesting video IDs strictly matching the target username
  while automatically dismissing login modals.
- Phase 2 (Stream Ingestion): Launches an isolated browser session with stealth
  automation to resolve unwatermarked CDN video streams (supporting both app-restricted
  TikTok Shop videos via web resolver and standard direct streams), saves each file
  strictly named <video_id>.mp4, and records downloaded IDs to the Download Archive.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import List, Optional, Set

from playwright.sync_api import BrowserContext, Page, Response, sync_playwright
from playwright_stealth import Stealth


# Configure plain text logging without decorative Unicode icons
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tiktok_playwright_crawler")


def extract_username_and_url(channel_input: str) -> tuple[str, str]:
    """Parse username and canonical profile URL from input string.

    Parameters
    ----------
    channel_input : str
        User handle (e.g. @dthaitu9999 or dthaitu9999) or profile URL.

    Returns
    -------
    tuple[str, str]
        A pair containing (username_without_at, canonical_profile_url).

    Raises
    ------
    ValueError
        If the channel input is empty or invalid.
    """
    cleaned = channel_input.strip()
    if not cleaned:
        raise ValueError("Channel input cannot be empty.")

    if cleaned.startswith("@"):
        username = cleaned.lstrip("@").split("/")[0].split("?")[0]
        return username, f"https://www.tiktok.com/@{username}"

    if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
        username = cleaned.split("/")[0].split("?")[0]
        return username, f"https://www.tiktok.com/@{username}"

    match = re.search(r"@([a-zA-Z0-9_\.\-]+)", cleaned)
    if match:
        username = match.group(1)
        return username, f"https://www.tiktok.com/@{username}"

    raise ValueError(f"Could not extract a '@username' handle from '{cleaned}'.")


class DownloadArchive:
    """Manages persistent record of successfully retrieved video identifiers."""

    def __init__(self, archive_path: str) -> None:
        """Initialize download archive.

        Parameters
        ----------
        archive_path : str
            File destination recording finished video IDs.
        """
        self.archive_path = archive_path
        self.seen_ids: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing recorded IDs from archive file."""
        if os.path.exists(self.archive_path):
            try:
                with open(self.archive_path, "r", encoding="utf-8") as f:
                    for line in f:
                        vid = line.strip()
                        if vid:
                            self.seen_ids.add(vid)
                logger.info("Loaded %d existing items from Download Archive.", len(self.seen_ids))
            except Exception as err:
                logger.warning("Could not read Download Archive at %s: %s", self.archive_path, err)

    def is_recorded(self, video_id: str) -> bool:
        """Check whether a video ID is already archived."""
        return video_id in self.seen_ids

    def record(self, video_id: str) -> None:
        """Record video ID to archive memory and disk."""
        if video_id not in self.seen_ids:
            self.seen_ids.add(video_id)
            os.makedirs(os.path.dirname(os.path.abspath(self.archive_path)), exist_ok=True)
            try:
                with open(self.archive_path, "a", encoding="utf-8") as f:
                    f.write(f"{video_id}\n")
            except Exception as err:
                logger.error("Failed to append %s to Download Archive: %s", video_id, err)


def harvest_channel_video_ids(
    playwright,
    username: str,
    canonical_url: str,
    max_videos: Optional[int] = None,
    archive: Optional[DownloadArchive] = None,
    headless: bool = True,
    scroll_pause: float = 2.5,
    max_scrolls: int = 60,
) -> List[str]:
    """Scroll channel profile in an isolated browser to collect video IDs.

    Parameters
    ----------
    playwright : Playwright
        Active Playwright manager instance.
    username : str
        Target creator username (without @).
    canonical_url : str
        Canonical web profile URL.
    max_videos : Optional[int]
        Target number of new un-archived video IDs to collect.
    archive : Optional[DownloadArchive]
        Archive to check existing downloads against.
    headless : bool
        Whether to run browser in headless mode.
    scroll_pause : float
        Pause duration between scroll pagination attempts.
    max_scrolls : int
        Maximum number of scroll pagination attempts.

    Returns
    -------
    List[str]
        Ordered list of discovered unique video IDs.
    """
    logger.info("Starting Phase 1: Catalogue Discovery for @%s...", username)
    browser = playwright.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    page = context.new_page()
    Stealth().apply_stealth_sync(page)

    logger.info("Navigating to creator profile: %s", canonical_url)
    try:
        page.goto(canonical_url, wait_until="networkidle", timeout=35000)
    except Exception as exc:
        logger.warning("Profile page navigation completed with: %s. Continuing...", exc)

    page.wait_for_timeout(3000)

    video_pattern = re.compile(rf"/@{re.escape(username)}/video/(\d+)", re.IGNORECASE)
    discovered_ids: List[str] = []
    seen_ids: Set[str] = set()

    def scan_page_for_videos() -> int:
        elements = page.query_selector_all("a[href*='/video/']")
        new_count = 0
        for el in elements:
            href = el.get_attribute("href") or ""
            match = video_pattern.search(href)
            if match:
                vid = match.group(1)
                if vid not in seen_ids:
                    seen_ids.add(vid)
                    discovered_ids.append(vid)
                    new_count += 1
        return new_count

    initial_new = scan_page_for_videos()
    logger.info("Initial profile load discovered %d videos.", initial_new)

    consecutive_no_new = 0

    for scroll_idx in range(1, max_scrolls + 1):
        # Calculate unarchived items if archive is provided
        if max_videos:
            unarchived = [v for v in discovered_ids if not (archive and archive.is_recorded(v))]
            if len(unarchived) >= max_videos:
                logger.info("Discovered requested quota of %d new videos. Halting scroll.", max_videos)
                break

        # Dismiss login popup by pressing Escape key
        page.keyboard.press("Escape")

        # Scroll down smoothly using mouse wheel
        page.mouse.wheel(0, 4500)
        page.wait_for_timeout(int(scroll_pause * 1000))

        # Close any lingering modal dialogs
        try:
            close_btn = page.query_selector("[data-e2e='modal-close-inner-button'], div[role='dialog'] [class*='close']")
            if close_btn:
                close_btn.click()
        except Exception:
            pass

        new_items = scan_page_for_videos()
        if new_items == 0:
            consecutive_no_new += 1
            if consecutive_no_new >= 6:
                logger.info("No additional videos discovered after 6 consecutive scrolls. Reached feed end.")
                break
        else:
            consecutive_no_new = 0
            logger.info("Scroll %d: +%d new videos (Total: %d)", scroll_idx, new_items, len(discovered_ids))

    browser.close()
    logger.info("Catalogue discovery complete. Found %d total videos for @%s.", len(discovered_ids), username)
    return discovered_ids


def download_stream_to_file(stream_url: str, target_path: str, min_size: int = 100000) -> bool:
    """Download binary stream from resolved CDN URL to disk.

    Parameters
    ----------
    stream_url : str
        Direct CDN download URL.
    target_path : str
        File destination path.
    min_size : int
        Minimum expected file size in bytes to consider valid.

    Returns
    -------
    bool
        True if successfully saved and passed size check, False otherwise.
    """
    temp_path = f"{target_path}.tmp"
    req = urllib.request.Request(
        stream_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://ssstik.io/",
            "Accept": "*/*",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp, open(temp_path, "wb") as out_file:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_file.write(chunk)

        if os.path.exists(temp_path) and os.path.getsize(temp_path) >= min_size:
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)
            return True

        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    except Exception as exc:
        logger.debug("Stream download failed for %s: %s", target_path, exc)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False


def resolve_video_ssstik(page: Page, username: str, video_id: str, target_path: str) -> bool:
    """Resolve and download unwatermarked video via ssstik automation.

    Parameters
    ----------
    page : Page
        Playwright page with stealth applied.
    username : str
        Target creator username.
    video_id : str
        Video identifier.
    target_path : str
        Output file path.

    Returns
    -------
    bool
        True if video was resolved and downloaded, False otherwise.
    """
    video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"

    try:
        page.goto("https://ssstik.io/en-1", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_selector("input#main_page_text", timeout=10000)
        page.fill("input#main_page_text", video_url)
        page.click("button#submit")

        # Wait for result overlay or download links
        page.wait_for_selector("a.download_link, .result_overlay", timeout=15000)
        page.wait_for_timeout(1000)

        # Dismiss any popup ads if present
        page.keyboard.press("Escape")

        download_url = None
        for a in page.query_selector_all("a.download_link"):
            href = a.get_attribute("href")
            text = (a.inner_text() or "").lower()
            if href and "without watermark" in text and "mp3" not in text:
                download_url = href
                break

        if download_url:
            success = download_stream_to_file(download_url, target_path)
            if success:
                return True

        logger.warning("ssstik did not yield a valid download URL for %s.", video_id)
        return False
    except Exception as exc:
        logger.warning("ssstik resolution error for %s: %s", video_id, exc)
        return False


def resolve_video_direct(context: BrowserContext, username: str, video_id: str, target_path: str) -> bool:
    """Fallback resolution: direct page navigation with CDN response interception.

    Excludes generic login modal animations (e.g. sf16-website-login).

    Parameters
    ----------
    context : BrowserContext
        Playwright browser context.
    username : str
        Creator username.
    video_id : str
        Video identifier.
    target_path : str
        Target file path.

    Returns
    -------
    bool
        True if video stream was intercepted and saved, False otherwise.
    """
    video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
    downloaded = False

    def on_response(response: Response) -> None:
        nonlocal downloaded
        if downloaded:
            return

        url = response.url
        ct = response.headers.get("content-type", "")

        # Exclude login background animations and marketing preview clips
        if "website-login" in url or "playback1.mp4" in url or "login_static" in url:
            return

        is_video_cdn = "webapp-prime" in url or "video/tos" in url or "tos-alisg" in url or "tos-maliva" in url or "tos-useast" in url
        if is_video_cdn and "video/mp4" in ct and response.status in (200, 206):
            try:
                body = response.body()
                if len(body) > 200000:
                    with open(target_path, "wb") as f:
                        f.write(body)
                    downloaded = True
            except Exception:
                pass

    page = context.new_page()
    Stealth().apply_stealth_sync(page)
    page.on("response", on_response)

    try:
        page.goto(video_url, wait_until="networkidle", timeout=25000)
    except Exception:
        pass

    start_wait = time.time()
    while not downloaded and (time.time() - start_wait) < 5.0:
        page.wait_for_timeout(500)

    try:
        page.close()
    except Exception:
        pass

    return downloaded


def download_single_video(
    page: Page,
    direct_context: BrowserContext,
    username: str,
    video_id: str,
    output_dir: str,
    archive: DownloadArchive,
) -> bool:
    """Download single unwatermarked video using primary resolver and fallback.

    Parameters
    ----------
    page : Page
        Dedicated stealth page for web resolver.
    direct_context : BrowserContext
        Context for direct TikTok fallback.
    username : str
        Target creator username.
    video_id : str
        Target video identifier.
    output_dir : str
        Output directory.
    archive : DownloadArchive
        Archive manager.

    Returns
    -------
    bool
        True if video was successfully downloaded and verified, False otherwise.
    """
    target_path = os.path.join(output_dir, f"{video_id}.mp4")

    # Skip if file already exists and is non-empty
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
        archive.record(video_id)
        logger.info("Video %s already exists on disk (%d bytes). Marked archived.", video_id, os.path.getsize(target_path))
        return True

    logger.info("Fetching Raw Video: %s", video_id)

    # Primary Tier: ssstik web resolver (handles App-Restricted / Shop videos)
    success = resolve_video_ssstik(page, username, video_id, target_path)

    # Secondary Tier: Direct TikTok page interception fallback
    if not success:
        logger.info("Primary resolver failed for %s. Attempting fallback direct interception...", video_id)
        success = resolve_video_direct(direct_context, username, video_id, target_path)

    if success and os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        file_size = os.path.getsize(target_path)
        archive.record(video_id)
        logger.info("SUCCESS: Saved %s (%d bytes)", target_path, file_size)
        return True

    logger.warning("FAILED: Unable to retrieve unwatermarked stream for %s.", video_id)
    return False


def crawl_channel(
    channel_input: str,
    output_dir: str = "data/crawled_videos",
    archive_file: Optional[str] = None,
    max_videos: Optional[int] = None,
    headless: bool = True,
    scroll_pause: float = 2.5,
    max_scrolls: int = 60,
    min_sleep: float = 1.0,
    max_sleep: float = 3.0,
) -> int:
    """Crawl a TikTok channel and download unwatermarked Raw Videos.

    Parameters
    ----------
    channel_input : str
        Channel handle or URL.
    output_dir : str
        Target destination directory for raw videos.
    archive_file : Optional[str]
        Path to download archive file.
    max_videos : Optional[int]
        Maximum number of videos to download.
    headless : bool
        Whether to run Chromium in headless mode.
    scroll_pause : float
        Pause interval between scroll actions.
    max_scrolls : int
        Maximum scroll pagination attempts.
    min_sleep : float
        Minimum polite delay between video downloads.
    max_sleep : float
        Maximum polite delay between video downloads.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for failure).
    """
    username, canonical_url = extract_username_and_url(channel_input)
    os.makedirs(output_dir, exist_ok=True)

    if not archive_file:
        archive_file = os.path.join(output_dir, ".download_archive.txt")

    archive = DownloadArchive(archive_file)

    logger.info("Creator Username: @%s", username)
    logger.info("Target Profile URL: %s", canonical_url)
    logger.info("Destination Directory: %s", os.path.abspath(output_dir))
    logger.info("Download Archive: %s", os.path.abspath(archive_file))
    if max_videos:
        logger.info("Download Limit: %d videos", max_videos)

    with sync_playwright() as playwright:
        # Phase 1: Harvest video IDs belonging strictly to this creator in an isolated session
        video_ids = harvest_channel_video_ids(
            playwright=playwright,
            username=username,
            canonical_url=canonical_url,
            max_videos=max_videos,
            archive=archive,
            headless=headless,
            scroll_pause=scroll_pause,
            max_scrolls=max_scrolls,
        )

        if not video_ids:
            logger.warning("No videos were discovered for @%s. Exiting.", username)
            return 1

        # Phase 2: Launch dedicated browser session for stream resolution
        logger.info("Starting Phase 2: Stream Ingestion for discovered videos...")
        download_browser = playwright.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        download_context = download_browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        resolver_page = download_context.new_page()
        Stealth().apply_stealth_sync(resolver_page)

        downloaded_count = 0
        skipped_count = 0
        failed_count = 0

        for vid in video_ids:
            if max_videos and downloaded_count >= max_videos:
                logger.info("Reached maximum download quota of %d videos. Finishing.", max_videos)
                break

            if archive.is_recorded(vid):
                logger.info("Video %s is already in Download Archive. Skipping.", vid)
                skipped_count += 1
                continue

            success = download_single_video(
                page=resolver_page,
                direct_context=download_context,
                username=username,
                video_id=vid,
                output_dir=output_dir,
                archive=archive,
            )
            if success:
                downloaded_count += 1
            else:
                failed_count += 1

            # Polite delay between items
            delay = random.uniform(min_sleep, max_sleep)
            time.sleep(delay)

        download_browser.close()

    logger.info("Crawl execution finished.")
    logger.info("Summary: %d downloaded, %d skipped (archived), %d failed.", downloaded_count, skipped_count, failed_count)
    return 0 if failed_count == 0 else 1


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Crawl unwatermarked Raw Videos from a TikTok channel with Playwright stealth automation."
    )
    parser.add_argument(
        "channel",
        type=str,
        help="TikTok channel URL (e.g. https://www.tiktok.com/@username) or handle (@username)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/crawled_videos",
        help="Target directory for downloaded raw videos (default: data/crawled_videos)",
    )
    parser.add_argument(
        "--archive-file",
        "-a",
        type=str,
        default=None,
        help="Path to Download Archive file for resume capability (default: <output_dir>/.download_archive.txt)",
    )
    parser.add_argument(
        "--max-videos",
        "-n",
        type=int,
        default=None,
        help="Maximum number of videos to download (default: None = all available)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in visible (headed) mode for debugging",
    )
    parser.add_argument(
        "--scroll-pause",
        type=float,
        default=2.5,
        help="Pause interval in seconds between scroll actions (default: 2.5)",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=60,
        help="Maximum scroll pagination attempts (default: 60)",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=1.0,
        help="Minimum polite delay between video downloads (default: 1.0)",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=3.0,
        help="Maximum polite delay between video downloads (default: 3.0)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_arguments()
    exit_code = crawl_channel(
        channel_input=args.channel,
        output_dir=args.output_dir,
        archive_file=args.archive_file,
        max_videos=args.max_videos,
        headless=not args.headed,
        scroll_pause=args.scroll_pause,
        max_scrolls=args.max_scrolls,
        min_sleep=args.min_sleep,
        max_sleep=args.max_sleep,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
