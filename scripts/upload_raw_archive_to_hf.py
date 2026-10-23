"""
Script to package and upload raw video parts and metadata to a private Hugging Face Dataset repository.

This script supports:
- Automatic private dataset repository creation on Hugging Face Hub.
- Per-Video Part archiving (.zip) to enable partitioned, resume-capable uploads.
- Skip-if-exists verification against the remote repository to handle interrupted transfers gracefully.
- Upload of canonical metadata tables (label-origin.csv, video_metadata.csv, outliers report).
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, get_token, login


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and upload video parts and metadata to Hugging Face Dataset repository."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g., 'username/harmful-video-archive').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face access token with write permission. Defaults to HF_TOKEN env var or cached token in ~/.cache/huggingface.",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=Path("data/video_parts"),
        help="Directory containing video partition subdirectories (part_01, part_02, ...).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing canonical metadata files (label-origin.csv, etc.).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/archives_staging"),
        help="Local staging directory where zipped video parts are generated.",
    )
    parser.add_argument(
        "--delete-local-archive-on-success",
        action="store_true",
        help="Delete local zip file immediately after its remote upload succeeds to save local disk space.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create repository as private (default: True).",
    )
    return parser.parse_args()


def zip_directory(source_dir: Path, target_zip: Path) -> None:
    """
    Compress a directory into a zip archive with deflate compression.
    Preserves relative paths under the directory name.
    """
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = target_zip.with_suffix(".tmp")
    
    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_STORED) as archive:
            # Using ZIP_STORED because MP4 files are already compressed (H.264/AAC);
            # compressing with DEFLATED wastes CPU cycles without significant byte reduction.
            for file_path in source_dir.rglob("*.mp4"):
                rel_path = file_path.relative_to(source_dir.parent)
                archive.write(file_path, arcname=str(rel_path))
        
        if target_zip.exists():
            target_zip.unlink()
        temp_zip.rename(target_zip)
    finally:
        if temp_zip.exists():
            temp_zip.unlink()


def get_remote_repo_files(api: HfApi, repo_id: str) -> List[str]:
    """Retrieve existing file list from the remote dataset repository to allow resume."""
    try:
        return api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return []


def main() -> None:
    args = parse_arguments()

    token = args.token or get_token()
    if not token:
        print("[ERROR] Hugging Face token is missing. Provide via --token, set HF_TOKEN environment variable, or login via huggingface-cli login.")
        sys.exit(1)

    login(token=token)
    api = HfApi(token=token)

    current_user = api.whoami().get("name", "unknown")
    print(f"[INFO] Authenticated as Hugging Face user: {current_user}")

    # 1. Initialize remote repository
    print(f"[INFO] Initializing dataset repository: {args.repo_id} (private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    remote_files = set(get_remote_repo_files(api, args.repo_id))
    print(f"[INFO] Existing files in remote repository: {len(remote_files)}")

    # 2. Upload metadata files first
    metadata_files = [
        "label-origin.csv",
        "video_metadata.csv",
        "video_outliers_report.csv",
    ]

    print("[INFO] Checking and uploading metadata files...")
    for filename in metadata_files:
        local_path = args.metadata_dir / filename
        remote_path = f"metadata/{filename}"
        if not local_path.exists():
            print(f"[WARNING] Metadata file not found locally, skipping: {local_path}")
            continue

        if remote_path in remote_files:
            print(f"[INFO] Metadata already uploaded, skipping: {remote_path}")
            continue

        print(f"[INFO] Uploading metadata: {local_path} -> {remote_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add metadata {filename}",
        )
        print(f"[SUCCESS] Uploaded {filename}")

    # 3. Process Video Parts
    if not args.parts_dir.exists():
        print(f"[ERROR] Video parts directory not found: {args.parts_dir}")
        sys.exit(1)

    part_dirs = sorted([d for d in args.parts_dir.iterdir() if d.is_dir() and d.name.startswith("part_")])
    if not part_dirs:
        print(f"[WARNING] No part_* directories found in {args.parts_dir}")
        return

    print(f"[INFO] Found {len(part_dirs)} video parts to process.")
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    for part_dir in part_dirs:
        part_name = part_dir.name
        remote_zip_name = f"videos/{part_name}.zip"

        if remote_zip_name in remote_files:
            print(f"[INFO] {remote_zip_name} already exists in remote repository. Skipping.")
            continue

        local_zip_path = args.staging_dir / f"{part_name}.zip"

        # Check if archive was already created locally
        if not local_zip_path.exists():
            print(f"[INFO] Archiving {part_name} into {local_zip_path}...")
            zip_directory(part_dir, local_zip_path)
            size_gb = local_zip_path.stat().st_size / (1024**3)
            print(f"[SUCCESS] Packaged {part_name}. Size: {size_gb:.2f} GB")
        else:
            print(f"[INFO] Found existing local archive: {local_zip_path}")

        print(f"[INFO] Uploading {local_zip_path.name} to {remote_zip_name} on Hugging Face...")
        api.upload_file(
            path_or_fileobj=str(local_zip_path),
            path_in_repo=remote_zip_name,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload {part_name} archive",
        )
        print(f"[SUCCESS] Upload completed for {part_name}")

        if args.delete_local_archive_on_success:
            print(f"[INFO] Deleting local archive {local_zip_path} to reclaim disk space.")
            local_zip_path.unlink()

    # Clean up staging directory if empty
    if args.staging_dir.exists() and not any(args.staging_dir.iterdir()):
        shutil.rmtree(args.staging_dir, ignore_errors=True)

    print("[SUCCESS] All video parts and metadata have been synchronized to Hugging Face.")


if __name__ == "__main__":
    main()
