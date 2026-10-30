"""
Script to package a new batch of videos into sequential parts and upload them to a specific branch on Hugging Face.

Key Features:
- Slices arbitrary video batches into standardized Video Part archives (e.g., part_14, part_15, part_16 of 500 items each).
- Merges new labels with existing canonical labels into an updated label-origin.csv on the target branch.
- Automatically creates the target Git branch (e.g., 'version-2') if it does not exist, branching from 'main'.
- Inherits all existing parts from 'main' via Git LFS deduplication (zero redundant re-uploads).
- Resume-capable: skips uploading parts that already exist on the target branch.
- Reclaims local disk space with --delete-local-archive-on-success.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
from huggingface_hub import HfApi, get_token, login


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and upload a new video batch to a specific Hugging Face branch."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to folder containing 'videos/' subdirectory and 'dataset.csv' (e.g., 'D:/Workspace/KLTN/consolidated_dataset_30_10').",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="KietTuannn/hierarchical-harmful-video-archive",
        help="Target Hugging Face repository ID.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="version-2",
        help="Target Git branch on Hugging Face (default: 'version-2').",
    )
    parser.add_argument(
        "--base-label-csv",
        type=Path,
        default=Path("data/label-origin.csv"),
        help="Path to existing canonical label CSV to merge with (default: 'data/label-origin.csv').",
    )
    parser.add_argument(
        "--start-part-num",
        type=int,
        default=14,
        help="Starting partition number for new video parts (default: 14).",
    )
    parser.add_argument(
        "--part-size",
        type=int,
        default=500,
        help="Number of videos per partition archive (default: 500).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/archives_staging"),
        help="Staging directory for packaging zip archives.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face access token with write permission.",
    )
    parser.add_argument(
        "--delete-local-archive-on-success",
        action="store_true",
        help="Delete temporary local zip file immediately after its remote upload succeeds.",
    )
    return parser.parse_args()


def resolve_auth_token(provided_token: Optional[str] = None) -> str:
    """Resolve active Hugging Face token from parameter, environment, or local cache."""
    token = provided_token or os.environ.get("HF_TOKEN", "") or get_token()
    if not token:
        print("[ERROR] Hugging Face token is missing. Provide via --token, set HF_TOKEN env var, or login via huggingface-cli.")
        sys.exit(1)
    return token


def ensure_branch_exists(api: HfApi, repo_id: str, branch: str) -> None:
    """Ensure remote branch exists; create off main if missing."""
    if branch == "main":
        return
    try:
        api.create_branch(repo_id=repo_id, branch=branch, repo_type="dataset", exist_ok=True)
        print(f"[INFO] Verified remote branch '{branch}' exists or was created.")
    except Exception as exc:
        print(f"[WARNING] Could not explicitly create branch '{branch}': {exc}. Proceeding with upload...")


def zip_file_list(files: List[Path], part_folder_name: str, target_zip: Path) -> None:
    """
    Compress a list of video files into a zip archive storing them under part_folder_name.
    Uses ZIP_STORED because MP4 files are already compressed.
    """
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = target_zip.with_suffix(".tmp")

    try:
        with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_STORED) as archive:
            for file_path in files:
                arcname = f"{part_folder_name}/{file_path.name}"
                archive.write(file_path, arcname=arcname)

        if target_zip.exists():
            target_zip.unlink()
        temp_zip.rename(target_zip)
    finally:
        if temp_zip.exists():
            temp_zip.unlink()


def main() -> None:
    args = parse_arguments()
    token = resolve_auth_token(args.token)

    login(token=token)
    api = HfApi(token=token)

    current_user = api.whoami().get("name", "unknown")
    print(f"[INFO] Authenticated as: {current_user}")
    print(f"[INFO] Target Repository: {args.repo_id}")
    print(f"[INFO] Target Branch: {args.branch}")

    # 1. Validate source paths
    video_dir = args.source_dir / "videos"
    new_csv_path = args.source_dir / "dataset.csv"

    if not video_dir.exists() or not video_dir.is_dir():
        print(f"[ERROR] Videos directory not found at: {video_dir}")
        sys.exit(1)
    if not new_csv_path.exists():
        print(f"[ERROR] New labels file not found at: {new_csv_path}")
        sys.exit(1)

    # 2. Ensure remote branch exists
    ensure_branch_exists(api, args.repo_id, args.branch)
    remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset", revision=args.branch))
    print(f"[INFO] Existing files in branch '{args.branch}': {len(remote_files)}")

    # 3. Read and merge labels
    print(f"[INFO] Reading new labels from: {new_csv_path}")
    new_df = pd.read_csv(new_csv_path)

    if args.base_label_csv.exists():
        print(f"[INFO] Merging with existing base labels from: {args.base_label_csv}")
        base_df = pd.read_csv(args.base_label_csv)
        print(f"[INFO] Base count: {len(base_df)}, New batch count: {len(new_df)}")
        merged_df = pd.concat([base_df, new_df], ignore_index=True).drop_duplicates(subset=["filename"], keep="last")
        print(f"[INFO] Consolidated dataset size: {len(merged_df)} rows")
    else:
        print("[WARNING] Base label CSV not found; using new batch labels only.")
        merged_df = new_df

    # Upload consolidated labels to target branch
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as tmp_file:
        tmp_csv_path = Path(tmp_file.name)
        merged_df.to_csv(tmp_csv_path, index=False)

    try:
        print(f"[INFO] Uploading consolidated label-origin.csv ({len(merged_df)} samples) to branch '{args.branch}'...")
        api.upload_file(
            path_or_fileobj=str(tmp_csv_path),
            path_in_repo="metadata/label-origin.csv",
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.branch,
            commit_message=f"Update label-origin.csv on {args.branch} (total {len(merged_df)} samples)",
        )
        print(f"[SUCCESS] Uploaded metadata/label-origin.csv to branch '{args.branch}'")
    finally:
        if tmp_csv_path.exists():
            tmp_csv_path.unlink()

    # 4. Resolve video file lists and slice into partitions
    video_filenames = new_df["filename"].astype(str).tolist()
    video_file_paths = []
    missing_files = []

    for name in video_filenames:
        p = video_dir / name
        if p.exists():
            video_file_paths.append(p)
        else:
            missing_files.append(name)

    if missing_files:
        print(f"[ERROR] Found {len(missing_files)} missing video files referenced in dataset.csv! Sample: {missing_files[:5]}")
        sys.exit(1)

    print(f"[INFO] Total verified videos to partition: {len(video_file_paths)}")
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    # Slice into chunks of args.part_size
    chunk_size = args.part_size
    chunks = [video_file_paths[i : i + chunk_size] for i in range(0, len(video_file_paths), chunk_size)]

    for idx, chunk in enumerate(chunks):
        part_num = args.start_part_num + idx
        part_name = f"part_{part_num:02d}"
        remote_zip_path = f"videos/{part_name}.zip"

        if remote_zip_path in remote_files:
            print(f"[INFO] {remote_zip_path} already exists on branch '{args.branch}'. Skipping.")
            continue

        local_zip_path = args.staging_dir / f"{part_name}.zip"

        if not local_zip_path.exists():
            print(f"\n[INFO] Archiving {len(chunk)} videos into {local_zip_path} (part folder: {part_name})...")
            zip_file_list(chunk, part_name, local_zip_path)
            size_gb = local_zip_path.stat().st_size / (1024**3)
            print(f"[SUCCESS] Packaged {part_name}. Size: {size_gb:.2f} GB")
        else:
            print(f"[INFO] Found existing local archive: {local_zip_path}")

        print(f"[INFO] Uploading {local_zip_path.name} to {remote_zip_path} on branch '{args.branch}'...")
        api.upload_file(
            path_or_fileobj=str(local_zip_path),
            path_in_repo=remote_zip_path,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.branch,
            commit_message=f"Add {part_name} archive to {args.branch} ({len(chunk)} videos)",
        )
        print(f"[SUCCESS] Upload completed for {part_name}")

        if args.delete_local_archive_on_success:
            print(f"[INFO] Deleting temporary local zip {local_zip_path} to reclaim disk space.")
            local_zip_path.unlink()

    # Clean up empty staging dir
    if args.staging_dir.exists() and not any(args.staging_dir.iterdir()):
        shutil.rmtree(args.staging_dir, ignore_errors=True)

    print(f"\n[SUCCESS] All {len(chunks)} new video parts and consolidated labels uploaded successfully to branch '{args.branch}'!")


if __name__ == "__main__":
    main()
