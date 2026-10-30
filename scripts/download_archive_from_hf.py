"""
Script and helper module to download and unpack video parts and metadata from Hugging Face Hub.

Designed for both local workstations and cloud notebook environments (Kaggle / Google Colab).
Supports:
- Selective download of individual video parts (e.g., --part part_01).
- Full dataset download (--all).
- Git branch selection (--branch) to pull experimental batches.
- Rolling extract-and-delete mode (--disk-safe) to stay well within Kaggle's 20GB disk limit.
- Modular Python functions for direct programmatic import in training notebooks.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, get_token, hf_hub_download, login


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract video parts and metadata from Hugging Face Dataset repository."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Source Hugging Face dataset repository ID (e.g., 'username/harmful-video-archive').",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch or revision to download from (default: 'main').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face access token. Defaults to HF_TOKEN env var or cached token in ~/.cache/huggingface.",
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Download a single specific part (e.g., 'part_01', 'part_14').",
    )
    parser.add_argument(
        "--parts",
        nargs="*",
        default=None,
        help="Download specific multiple parts (e.g., --parts part_01 part_02).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all video parts found in the repository branch.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only canonical metadata files (CSV tables).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data"),
        help="Target base directory to save downloaded and extracted data (default: 'data').",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=True,
        help="Extract video zip archives after downloading (default: True).",
    )
    parser.add_argument(
        "--disk-safe",
        action="store_true",
        help="Immediately delete each downloaded zip file after extraction to preserve disk space (crucial for Kaggle 20GB limit).",
    )
    return parser.parse_args()


def resolve_auth_token(provided_token: Optional[str] = None) -> str:
    """Resolve active Hugging Face token from parameter, environment, or local cache."""
    token = provided_token or os.environ.get("HF_TOKEN", "") or get_token()
    if not token:
        print("[ERROR] Hugging Face token is missing. Provide via --token, set HF_TOKEN environment variable, or login via huggingface-cli login.")
        sys.exit(1)
    return token


def list_remote_parts(api: HfApi, repo_id: str, branch: str) -> List[str]:
    """Retrieve sorted list of all video part archive names available in the repo branch."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=branch)
    parts = []
    for file_path in files:
        if file_path.startswith("videos/") and file_path.endswith(".zip"):
            stem = Path(file_path).stem  # e.g., 'part_01'
            parts.append(stem)
    return sorted(parts)


def download_metadata(
    repo_id: str,
    target_dir: Path,
    branch: str = "main",
    token: Optional[str] = None,
) -> List[Path]:
    """
    Download canonical metadata files from the dataset repository.
    """
    resolved_token = resolve_auth_token(token)
    api = HfApi(token=resolved_token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=branch)
    metadata_files = [f for f in files if f.startswith("metadata/")]

    downloaded = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for remote_path in metadata_files:
        filename = Path(remote_path).name
        print(f"[INFO] Downloading metadata: {remote_path} -> {target_dir / filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=branch,
            token=resolved_token,
            local_dir=str(target_dir / ".hf_cache"),
        )
        dest_path = target_dir / filename
        shutil.copy2(local_path, dest_path)
        downloaded.append(dest_path)
        print(f"[SUCCESS] Saved metadata: {dest_path}")

    # Clean up temporary download cache folder if exists
    temp_cache = target_dir / ".hf_cache"
    if temp_cache.exists():
        shutil.rmtree(temp_cache, ignore_errors=True)

    return downloaded


def download_and_extract_video_part(
    repo_id: str,
    part_name: str,
    target_video_parts_dir: Path,
    branch: str = "main",
    token: Optional[str] = None,
    extract: bool = True,
    disk_safe: bool = False,
) -> Path:
    """
    Download an individual video part zip file, optionally extract into target directory,
    and optionally delete the zip file to save disk space.
    """
    resolved_token = resolve_auth_token(token)
    remote_filename = f"videos/{part_name}.zip"

    print(f"[INFO] Downloading {remote_filename} (branch: '{branch}')...")
    zip_download_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_filename,
        repo_type="dataset",
        revision=branch,
        token=resolved_token,
    )
    print(f"[SUCCESS] Downloaded: {zip_download_path}")

    target_video_parts_dir.mkdir(parents=True, exist_ok=True)

    if extract:
        print(f"[INFO] Extracting {part_name} into {target_video_parts_dir}...")
        with zipfile.ZipFile(zip_download_path, "r") as archive:
            archive.extractall(target_video_parts_dir)
        print(f"[SUCCESS] Extracted {part_name}.")

        if disk_safe:
            # Delete local cached file to free up disk space immediately
            try:
                os.remove(zip_download_path)
                print(f"[INFO] Disk-safe mode: Removed temporary archive {zip_download_path}")
            except OSError as err:
                print(f"[WARNING] Could not delete cached archive: {err}")

    return Path(zip_download_path)


def download_all_video_parts(
    repo_id: str,
    target_video_parts_dir: Path,
    branch: str = "main",
    token: Optional[str] = None,
    extract: bool = True,
    disk_safe: bool = True,
) -> None:
    """
    Download and sequentially process all video parts in rolling fashion.
    """
    resolved_token = resolve_auth_token(token)
    api = HfApi(token=resolved_token)
    available_parts = list_remote_parts(api, repo_id, branch)

    if not available_parts:
        print(f"[WARNING] No video parts found in repository '{repo_id}' under branch '{branch}'.")
        return

    print(f"[INFO] Found {len(available_parts)} parts to download: {available_parts}")

    for idx, part_name in enumerate(available_parts, start=1):
        print(f"\n[INFO] === Processing Part {idx}/{len(available_parts)}: {part_name} ===")
        download_and_extract_video_part(
            repo_id=repo_id,
            part_name=part_name,
            target_video_parts_dir=target_video_parts_dir,
            branch=branch,
            token=resolved_token,
            extract=extract,
            disk_safe=disk_safe,
        )

    print(f"\n[SUCCESS] Completed downloading all {len(available_parts)} parts.")


def main() -> None:
    args = parse_arguments()
    token = resolve_auth_token(args.token)

    login(token=token)
    api = HfApi(token=token)

    # 1. Always ensure metadata is downloaded first if requested or in full run
    if args.metadata_only:
        download_metadata(repo_id=args.repo_id, target_dir=args.target_dir, branch=args.branch, token=token)
        return

    # 2. Determine target parts
    target_parts_dir = args.target_dir / "video_parts"

    parts_to_download: List[str] = []
    if args.part:
        parts_to_download = [args.part]
    elif args.parts:
        parts_to_download = args.parts
    elif args.all:
        download_metadata(repo_id=args.repo_id, target_dir=args.target_dir, branch=args.branch, token=token)
        download_all_video_parts(
            repo_id=args.repo_id,
            target_video_parts_dir=target_parts_dir,
            branch=args.branch,
            token=token,
            extract=args.extract,
            disk_safe=args.disk_safe,
        )
        return
    else:
        print("[ERROR] Please specify --part, --parts, --all, or --metadata-only.")
        sys.exit(1)

    # Download designated subset of parts
    for part_name in parts_to_download:
        download_and_extract_video_part(
            repo_id=args.repo_id,
            part_name=part_name,
            target_video_parts_dir=target_parts_dir,
            branch=args.branch,
            token=token,
            extract=args.extract,
            disk_safe=args.disk_safe,
        )


if __name__ == "__main__":
    main()
