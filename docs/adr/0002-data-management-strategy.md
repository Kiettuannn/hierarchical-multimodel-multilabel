# 0002 Data Management Strategy via Tiered Hugging Face Hub Repositories

## Status
Accepted

## Context
The repository manages approximately 27 GB of raw video files across 13 `Video Part` partitions alongside extracted textual data, tabular annotations, and multimodal embeddings (Vision, ASR, OCR, CLAP). Feature extraction and model training are executed primarily on Cloud GPU environments (Kaggle and Google Colab) rather than local workstations, while developers require zero-maintenance infrastructure without self-hosting overhead.

## Decision
We adopt a tiered remote dataset strategy:
1. **Raw Video Archive on Hugging Face Hub**: Raw `.mp4` video files partitioned by `Video Part` (`part_01` through `part_13`) are compressed into individual archives (`.zip`) and stored in a private Hugging Face dataset repository. This serves as the authoritative, long-term cold storage archive.
2. **Metadata and Label Mirroring**: `Label Origin` (`label-origin.csv`), `Video Metadata` (`video_metadata.csv`), and outliers reports are tracked in Git and mirrored in the Hugging Face repository.
3. **Feature Registry on Cloud Training Platforms**: Multimodal continuous embeddings (Vision, ASR, OCR, CLAP) are generated directly within cloud GPU environments (e.g., Kaggle notebooks) and saved directly into platform-native datasets (e.g., Kaggle Datasets). This completely avoids uploading multi-gigabyte intermediate embeddings from local machines over residential uplinks.
4. **Branch-based Experimental Staging**: Experimental or unverified video batches (e.g., `part_14`) are published to dedicated Git branches (e.g., `experiment-batch-02`) on the Hugging Face repository. This isolates research trials from the canonical `main` ground-truth dataset and allows clean verification before merging via Pull Requests.

## Considered Options
- **MinIO + DVC**: Rejected due to network egress bottlenecks from local machines to Cloud GPUs (Colab/Kaggle), complex credential sharing across distributed nodes, and operational maintenance overhead of hosting local storage servers.
- **Local Embedding Upload to Hugging Face**: Rejected because embeddings can be generated faster and cheaper on cloud GPU nodes and preserved directly in Kaggle/Colab environments without local bandwidth consumption.
- **Single Branch Staging**: Rejected because committing untested videos directly to `main` contaminates the canonical dataset and risks training leakage.

## Consequences
- Cloud GPU notebooks can pull individual `Video Part` archives or full datasets on-demand from Hugging Face via `huggingface_hub` using token authentication (`HF_TOKEN`).
- Zero local DevOps maintenance for storage daemons or disk volume arrays.
- Fast, bandwidth-efficient pipeline decoupled between raw cold storage (Hugging Face) and cloud-computed features (Kaggle Datasets).
- Clean experiment isolation via Git branches without repository duplication.
