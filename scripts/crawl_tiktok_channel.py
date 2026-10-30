"""TikTok Channel Video Crawler.

This script crawls and downloads unwatermarked raw videos from a specified
TikTok channel URL or user handle. Video files are saved strictly named by
their platform video identifier (<video_id>.mp4). An idempotent download
archive tracks finished items to support seamless resumption across runs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget
from yt_dlp.utils import MaxDownloadsReached


# Configure clean terminal logging without decorative Unicode icons
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tiktok_crawler")


def normalize_channel_url(raw_input: str) -> str:
    """Normalize user input into a canonical TikTok channel URL.

    Handles full profile URLs, shortened share links (e.g., vt.tiktok.com),
    prefixed handles (@username), and bare usernames.

    Parameters
    ----------
    raw_input : str
        Input string representing the channel URL or user handle.

    Returns
    -------
    str
        Standardized channel URL formatted as https://www.tiktok.com/@username.

    Raises
    ------
    ValueError
        If the channel input cannot be resolved into a valid URL.
    """
    cleaned = raw_input.strip()
    if not cleaned:
        raise ValueError("Channel input cannot be empty.")

    # Convert bare username or @username to canonical web profile URL
    if cleaned.startswith("@"):
        return f"https://www.tiktok.com/{cleaned}"
    if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
        return f"https://www.tiktok.com/@{cleaned}"

    # Resolve mobile shortened share links by following HTTP redirection
    parsed = urllib.parse.urlparse(cleaned)
    if "tiktok.com" in parsed.netloc and ("vt." in parsed.netloc or "vm." in parsed.netloc):
        logger.info("Resolving shortened share link: %s", cleaned)
        req = urllib.request.Request(
            cleaned,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                cleaned = response.geturl()
                logger.info("Resolved target profile URL: %s", cleaned)
        except Exception as exc:
            logger.warning(
                "Failed to follow redirection for %s: %s. Using URL as-is.",
                cleaned,
                exc,
            )

    # Standardize scheme and domain if needed
    if not re.search(r"@[\w\.\-]+", cleaned):
        logger.warning(
            "Target URL '%s' does not appear to contain a '@username' handle. "
            "Proceeding with the URL directly.",
            cleaned,
        )

    return cleaned


class MetadataCollector:
    """Collects technical metadata for downloaded videos to build optional manifests."""

    def __init__(self, manifest_path: Optional[str] = None) -> None:
        """Initialize metadata collector.

        Parameters
        ----------
        manifest_path : Optional[str]
            Path to the output CSV manifest file. If None, metadata is not recorded.
        """
        self.manifest_path = manifest_path
        self.records: List[Dict[str, Any]] = []
        self._seen_ids: set[str] = set()

        if self.manifest_path and os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vid = row.get("video_id")
                        if vid:
                            self._seen_ids.add(vid)
                            self.records.append(row)
            except Exception as err:
                logger.warning("Could not pre-read existing manifest at %s: %s", self.manifest_path, err)

    def record_video(self, info: Dict[str, Any]) -> None:
        """Record technical video attributes from yt-dlp extracted info dictionary.

        Parameters
        ----------
        info : Dict[str, Any]
            Extracted info dictionary from yt-dlp.
        """
        if not self.manifest_path:
            return

        video_id = str(info.get("id") or "")
        if not video_id or video_id in self._seen_ids:
            return

        record = {
            "video_id": video_id,
            "filename": f"{video_id}.mp4",
            "title": info.get("title") or "",
            "uploader": info.get("uploader") or info.get("channel") or "",
            "upload_date": info.get("upload_date") or "",
            "duration": info.get("duration") or "",
            "webpage_url": info.get("webpage_url") or "",
            "view_count": info.get("view_count") or "",
            "like_count": info.get("like_count") or "",
        }
        self.records.append(record)
        self._seen_ids.add(video_id)
        self._flush()

    def _flush(self) -> None:
        """Flush in-memory records to disk as a CSV manifest."""
        if not self.manifest_path:
            return

        fieldnames = [
            "video_id",
            "filename",
            "title",
            "uploader",
            "upload_date",
            "duration",
            "webpage_url",
            "view_count",
            "like_count",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)


class CrawlerLogger:
    """Custom logger routing messages from YoutubeDL and recording failure diagnostics."""

    def __init__(self, failed_log_path: str, verbose: bool = False) -> None:
        """Initialize crawler logger.

        Parameters
        ----------
        failed_log_path : str
            File destination to append failed URLs.
        verbose : bool
            Whether to output debug messages.
        """
        self.failed_log_path = failed_log_path
        self.verbose = verbose
        self._warned_about_shop = False

    def debug(self, msg: str) -> None:
        """Process debug message."""
        if self.verbose:
            logger.debug(msg)

    def info(self, msg: str) -> None:
        """Process info message."""
        if self.verbose:
            logger.info(msg)

    def warning(self, msg: str) -> None:
        """Process warning message."""
        logger.warning(msg)

    def error(self, msg: str) -> None:
        """Process error message and diagnose specific failure types."""
        logger.error(msg)
        # Check for TikTok Shop / app restriction indicators
        is_shop_error = (
            "tiktok shop" in msg.lower()
            or "403: forbidden" in msg.lower()
            or "only audio" in msg.lower()
        )
        if is_shop_error and not self._warned_about_shop:
            self._warned_about_shop = True
            logger.warning(
                "DIAGNOSTIC: Video appears restricted by TikTok Shop or requires authenticated session cookies. "
                "Try passing '--cookies-from-browser chrome' (or edge/firefox) or '--cookies cookies.txt' "
                "to allow yt-dlp to access authenticated desktop sessions."
            )

        # Log failed video URL to file for subsequent inspection
        match = re.search(r"https?://(?:www\.)?tiktok\.com/[^\s]+", msg)
        if match:
            url = match.group(0).rstrip(":")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.failed_log_path)), exist_ok=True)
                with open(self.failed_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{url}\n")
            except Exception:
                pass


def build_ydl_options(
    output_dir: str,
    archive_file: str,
    failed_log_path: str,
    min_sleep: float,
    max_sleep: float,
    sleep_requests: float,
    max_videos: Optional[int],
    date_after: Optional[str],
    impersonate_client: str,
    cookies_file: Optional[str],
    cookies_browser: Optional[str],
    verbose: bool,
    metadata_collector: Optional[MetadataCollector],
) -> Dict[str, Any]:
    """Construct configuration options dictionary for YoutubeDL.

    Parameters
    ----------
    output_dir : str
        Directory to place downloaded raw mp4 files.
    archive_file : str
        File path tracking successfully downloaded video IDs.
    failed_log_path : str
        File path to record failed video URLs.
    min_sleep : float
        Minimum sleep interval in seconds between video downloads.
    max_sleep : float
        Maximum sleep interval in seconds between video downloads.
    sleep_requests : float
        Sleep interval between individual API/page requests to avoid 403 blocks.
    max_videos : Optional[int]
        Cap on the number of videos to download.
    date_after : Optional[str]
        Date threshold in YYYYMMDD format.
    impersonate_client : str
        Client target string for TLS fingerprint spoofing via curl_cffi.
    cookies_file : Optional[str]
        Path to Netscape-format cookie file.
    cookies_browser : Optional[str]
        Browser name to extract cookies from directly.
    verbose : bool
        Whether to show debug logging.
    metadata_collector : Optional[MetadataCollector]
        Optional hook to capture video metadata.

    Returns
    -------
    Dict[str, Any]
        Option mapping consumed by yt_dlp.YoutubeDL.
    """
    # Enforce naming contract: raw video named strictly by platform video ID
    outtmpl_pattern = os.path.join(output_dir, "%(id)s.%(ext)s")

    # Set up TLS client impersonation to bypass TikTok HTTP 403 / anti-bot blocks
    target = ImpersonateTarget.from_str(impersonate_client)

    ydl_opts: Dict[str, Any] = {
        "outtmpl": outtmpl_pattern,
        "download_archive": archive_file,
        # Prefer direct unwatermarked streams (best video stream without overlay)
        "format": "bv*+ba/b",
        # Impersonate modern desktop browser TLS fingerprint
        "impersonate": target,
        # Sleep intervals between downloads to maintain polite crawling rate
        "sleep_interval": min_sleep,
        "max_sleep_interval": max_sleep,
        # Sleep interval between pagination requests
        "sleep_interval_requests": sleep_requests,
        # Automatic retry logic on intermittent network failures or HTTP 429
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep": 5.0,
        # Exclude auxiliary sidecar files (subtitles, thumbnails, info json)
        "writesubtitles": False,
        "writethumbnail": False,
        "writeinfojson": False,
        "quiet": not verbose,
        "no_warnings": not verbose,
        # Ignore individual failed items so full channel crawling continues uninterrupted
        "ignoreerrors": True,
        "logger": CrawlerLogger(failed_log_path=failed_log_path, verbose=verbose),
    }

    if max_videos and max_videos > 0:
        ydl_opts["max_downloads"] = max_videos

    if date_after:
        # Standardize date format YYYY-MM-DD into YYYYMMDD
        sanitized_date = date_after.replace("-", "")
        ydl_opts["daterange"] = yt_dlp.utils.DateRange(start=sanitized_date)

    if cookies_file:
        # Sanitize cookies to filter out expired or domain-locked WAF tokens (_waftokenid)
        if os.path.exists(cookies_file):
            try:
                with open(cookies_file, "r", encoding="utf-8", errors="ignore") as f:
                    raw_lines = f.readlines()
                if any("_waftokenid" in line for line in raw_lines):
                    sanitized_path = os.path.join(output_dir, ".sanitized_cookies.txt")
                    os.makedirs(output_dir, exist_ok=True)
                    with open(sanitized_path, "w", encoding="utf-8") as f:
                        f.writelines([line for line in raw_lines if "_waftokenid" not in line])
                    ydl_opts["cookiefile"] = sanitized_path
                else:
                    ydl_opts["cookiefile"] = cookies_file
            except Exception:
                ydl_opts["cookiefile"] = cookies_file
        else:
            ydl_opts["cookiefile"] = cookies_file
    elif cookies_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_browser, None, None, None)

    if metadata_collector and metadata_collector.manifest_path:
        def post_download_hook(data: Dict[str, Any]) -> None:
            if data.get("status") == "finished":
                info_dict = data.get("info_dict") or {}
                metadata_collector.record_video(info_dict)

        ydl_opts["progress_hooks"] = [post_download_hook]

    return ydl_opts


def crawl_tiktok_channel(
    channel_url: str,
    output_dir: str = "data/crawled_videos",
    archive_file: Optional[str] = None,
    min_sleep: float = 2.0,
    max_sleep: float = 5.0,
    sleep_requests: float = 1.0,
    max_videos: Optional[int] = None,
    date_after: Optional[str] = None,
    save_metadata: bool = False,
    impersonate_client: str = "chrome",
    cookies_file: Optional[str] = None,
    cookies_browser: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Execute crawl of a TikTok channel and download unwatermarked raw videos.

    Parameters
    ----------
    channel_url : str
        Channel web URL or user handle.
    output_dir : str
        Directory where raw video files will be stored.
    archive_file : Optional[str]
        Path to download archive file. If None, defaults to .download_archive.txt
        inside output_dir.
    min_sleep : float
        Minimum sleep seconds between video downloads.
    max_sleep : float
        Maximum sleep seconds between video downloads.
    sleep_requests : float
        Sleep seconds between pagination requests.
    max_videos : Optional[int]
        Optional maximum number of videos to download.
    date_after : Optional[str]
        Optional cutoff date (YYYYMMDD or YYYY-MM-DD) to skip older videos.
    save_metadata : bool
        If True, exports a CSV manifest (metadata_manifest.csv) in output_dir.
    impersonate_client : str
        Browser identity to impersonate with TLS fingerprinting (default: chrome).
    cookies_file : Optional[str]
        Path to Netscape-format cookie file.
    cookies_browser : Optional[str]
        Browser name to load session cookies from.
    verbose : bool
        Enable verbose extractor logs.

    Returns
    -------
    int
        Exit status code (0 for success, non-zero for failure).
    """
    normalized_url = normalize_channel_url(channel_url)
    os.makedirs(output_dir, exist_ok=True)

    if not archive_file:
        archive_file = os.path.join(output_dir, ".download_archive.txt")

    failed_log_path = os.path.join(output_dir, "failed_videos.txt")
    manifest_path = os.path.join(output_dir, "metadata_manifest.csv") if save_metadata else None
    metadata_collector = MetadataCollector(manifest_path) if save_metadata else None

    logger.info("Target channel URL: %s", normalized_url)
    logger.info("Destination directory: %s", os.path.abspath(output_dir))
    logger.info("Download archive path: %s", os.path.abspath(archive_file))
    if max_videos:
        logger.info("Maximum download limit: %d videos", max_videos)
    if date_after:
        logger.info("Date threshold (after): %s", date_after)
    if cookies_browser:
        logger.info("Loading session cookies from browser: %s", cookies_browser)
    elif cookies_file:
        logger.info("Loading session cookies from file: %s", cookies_file)
    if save_metadata:
        logger.info("Metadata manifest target: %s", os.path.abspath(manifest_path))  # type: ignore[arg-type]
    else:
        logger.info("Mode: Strictly raw videos only (no sidecar metadata files).")

    ydl_opts = build_ydl_options(
        output_dir=output_dir,
        archive_file=archive_file,
        failed_log_path=failed_log_path,
        min_sleep=min_sleep,
        max_sleep=max_sleep,
        sleep_requests=sleep_requests,
        max_videos=max_videos,
        date_after=date_after,
        impersonate_client=impersonate_client,
        cookies_file=cookies_file,
        cookies_browser=cookies_browser,
        verbose=verbose,
        metadata_collector=metadata_collector,
    )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Starting crawler process...")
            exit_code = ydl.download([normalized_url])
            if exit_code == 0:
                logger.info("SUCCESS: Crawl and download completed successfully.")
            else:
                logger.warning("Crawler finished with skipped or failed items (exit code %d).", exit_code)
                if os.path.exists(failed_log_path):
                    logger.info("A list of unextractable/restricted URLs was recorded at: %s", failed_log_path)
            return exit_code
    except MaxDownloadsReached:
        logger.info("SUCCESS: Reached maximum requested video downloads limit.")
        return 0
    except Exception as exc:
        logger.error("Crawl encountered an unhandled exception: %s", exc)
        return 1


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the TikTok crawler CLI.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Crawl unwatermarked raw videos from a TikTok channel URL and save "
            "each sample named strictly by its video ID (<video_id>.mp4)."
        )
    )
    parser.add_argument(
        "channel",
        type=str,
        help="TikTok channel URL (e.g. https://www.tiktok.com/@username) or user handle (@username)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/crawled_videos",
        help="Target directory to store downloaded raw videos (default: data/crawled_videos)",
    )
    parser.add_argument(
        "--archive-file",
        "-a",
        type=str,
        default=None,
        help="Path to download archive file for resume capability (default: <output_dir>/.download_archive.txt)",
    )
    parser.add_argument(
        "--max-videos",
        "-n",
        type=int,
        default=None,
        help="Maximum number of videos to download (default: None = all videos)",
    )
    parser.add_argument(
        "--date-after",
        type=str,
        default=None,
        help="Filter videos uploaded on or after date (format: YYYYMMDD or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=2.0,
        help="Minimum polite sleep seconds between downloads (default: 2.0)",
    )
    parser.add_argument(
        "--max-sleep",
        type=float,
        default=5.0,
        help="Maximum polite sleep seconds between downloads (default: 5.0)",
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=1.0,
        help="Polite sleep seconds between pagination requests (default: 1.0)",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Export metadata_manifest.csv in the output directory (default: False, raw videos only)",
    )
    parser.add_argument(
        "--impersonate",
        type=str,
        default="chrome",
        help="Client TLS fingerprint target via curl_cffi (default: chrome)",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to Netscape-format cookies.txt file",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Browser name to load cookies from (e.g., chrome, firefox, edge)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output from yt-dlp extractor",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_arguments()
    code = crawl_tiktok_channel(
        channel_url=args.channel,
        output_dir=args.output_dir,
        archive_file=args.archive_file,
        min_sleep=args.min_sleep,
        max_sleep=args.max_sleep,
        sleep_requests=args.sleep_requests,
        max_videos=args.max_videos,
        date_after=args.date_after,
        save_metadata=args.save_metadata,
        impersonate_client=args.impersonate,
        cookies_file=args.cookies,
        cookies_browser=args.cookies_from_browser,
        verbose=args.verbose,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
