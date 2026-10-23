# Hierarchical Multimodal Multilabel Video Classification

This repository classifies multimodal short-form videos into hierarchical harmfulness categories using visual, audio, and textual signals.

## Language

**Harm Category**:
One of the seven predefined harmful taxonomy labels: `information_harm`, `sexual_harm`, `psychological_harm`, `hate_harassment_harm`, `clickbait_harm`, `addictive_harm`, or `physical_harm`.
_Avoid_: Class, target, tag, label type

**Normal Content**:
A video sample annotated with zero harm categories, indicating non-violating content.
_Avoid_: Clean video, negative sample, safe video, benign content

**Harmful Video**:
A video sample annotated with at least one harm category.
_Avoid_: Positive sample, toxic video, violation sample

**Video Part**:
A partition subdirectory within `data/video_parts/` (`part_01` through `part_13`) storing raw `.mp4` video files.
_Avoid_: Chunk, batch, shard, subfolder

**Label Origin**:
The canonical ground-truth CSV file (`data/label-origin.csv`) defining the dataset samples and their seven binary harm annotations.
_Avoid_: Raw labels, annotation table, source CSV

**Video Metadata**:
Extracted technical attributes of a video file, specifically duration, resolution width and height, frame rate, file size, and file read validity.
_Avoid_: Video stats, video properties, video info

**Raw Video Archive**:
The cold-storage collection of raw `.mp4` video files partitioned into Video Parts and archived on a remote registry.
_Avoid_: Video dump, raw bucket, video blob

**Feature Registry**:
The versioned storage repository holding precomputed multimodal embeddings (Vision, ASR, OCR, CLAP) and tabular prior signals keyed by video filename.
_Avoid_: Embedding store, feature dump

**Data Split**:
The reproducible partition assignment of dataset items into train, validation, and test subsets preserved via split configuration tables.
_Avoid_: Data fold, train split set, dataset partition
