# YouTube Summarizer
Summarize YouTube videos

## How it works
1. Input how many days' worth of videos (N) you want to summarize.
2. Input which channel C to summarize.
3. Download N days of video audio from channel C.
4. Detect language and perform text-to-speech on those videos.
5. Generate multiple summaries:
- A summary for each video
- An overall summary for all videos.

## Setup
```sh
mkdir .venv
python -m venv ./.venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install yt-dlp==2024.5.27
```
