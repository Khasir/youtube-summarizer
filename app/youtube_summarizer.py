"""
Example of use::

    >>> title, summary = summarize_video('https://www.youtube.com/watch?v=0JouTsMQsEA', token_batch_size=100)
    [youtube] Extracting URL: https://www.youtube.com/watch?v=0JouTsMQsEA
    [youtube] 0JouTsMQsEA: Downloading webpage
    [youtube] 0JouTsMQsEA: Downloading ios player API JSON
    [youtube] 0JouTsMQsEA: Downloading player 03dbdfab
    WARNING: [youtube] 0JouTsMQsEA: nsig extraction failed: You may experience throttling for some formats
            n = vza2ivCfRzWGbfkrQ ; player = https://www.youtube.com/s/player/03dbdfab/player_ias.vflset/en_US/base.js
    WARNING: [youtube] 0JouTsMQsEA: nsig extraction failed: You may experience throttling for some formats
            n = iumHJFtfqPnJSV_0a ; player = https://www.youtube.com/s/player/03dbdfab/player_ias.vflset/en_US/base.js
    [youtube] 0JouTsMQsEA: Downloading m3u8 information
    >>> print("Title:", title)
    Title: RollerCoaster Tycoon was the last of its kind.
    >>> print("Summary:", summary)
    Summary: Chris Sawyer was a 15-year-old programmer when he made the first Miner. Locomotion was a spiritual successor to Transport Tycoon. In 2002 it had sold over 4 million copies and Chris had made around $30 million in royalties.
"""

import datetime
import math
import unicodedata

import requests
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from yt_dlp import YoutubeDL


assert torch.cuda.is_available()

model_name = "facebook/bart-large-cnn"
print(f"Initializing {model_name}...")
bart = BartForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2"
    # device_map='cuda',
    ignore_mismatched_sizes=True,
).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Initialized.")


def get_video_metadata(url: str, params: dict = None):
    if params is None:
        params = {
            # "progress": True,
        }
    with YoutubeDL(params=params) as ydl:
        info = ydl.extract_info(url, download=False)
        return info


def list_most_recent_videos(days: int, channel: str):
    today = datetime.datetime.today()
    diff = datetime.timedelta(days=-days)
    target = today + diff
    target = target.strftime("%Y%m%d")
    params = {
        "dateafter": target, 
        "extractor-args": {"youtubetab": {"approximate_date": ['']}},
        "progress": True,
        # "write-pages": True,
    }

    with YoutubeDL(params=params) as ydl:
        info = ydl.extract_info(channel, download=False)
        return info


def get_caption_file_url(metadata: dict, lang: str = 'en'):
    found = None
    if lang == 'en':
        lang_codes = 'en-orig', 'en-US', 'en-GB', 'en'
    else:
        lang_codes = lang,

    # Prefer human-typed subtitles over automatic captions
    subtitles = metadata.get('subtitles')
    if subtitles:
        for code in lang_codes:
            if code in subtitles:
                found = subtitles[code]
                break
    if not found:
        auto_captions = metadata.get('automatic_captions')
        if auto_captions:
            for code in lang_codes:
                if code in auto_captions:
                    found = auto_captions[code]
                    break
    
    if not found:
        return None
    
    for entry in found:
        if entry['ext'] == 'json3':
            return entry['url']
    
    # Hopefully this shouldn't happen
    return found[0]['url']


def get_text(captions: dict) -> str:
    text = []
    for event in captions['events']:
        if 'segs' in event:
            for segment in event['segs']:
                text.append(segment['utf8'])
    return ''.join(text)


def summarize(text_batches: list) -> str:
    if isinstance(text_batches, str):
        text_batches = [text_batches]
    inputs = tokenizer(text_batches, max_length=1024, return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to('cuda')
    summary_ids = bart.generate(inputs['input_ids'], num_beams=2, min_length=0, max_length=100)
    result = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return result


def summarize_long_text(text: str, token_batch_size: int = 1024, return_partial_summaries=False) -> str:
    # Loop while text length exceeds token batch size
    while True:
        # Get total approximate token length of the text
        text = unicodedata.normalize('NFKC', text)
        text = ' '.join(text.split())
        whole_inputs = tokenizer([text], return_tensors='pt', truncation=False, add_special_tokens=False)
        num_tokens = len(whole_inputs['input_ids'][0])
        num_batches = math.ceil(num_tokens / token_batch_size)

        # Leave loop if text is short enough
        if num_batches == 1:
            break

        # Batchify text
        text_batches = []
        for i in range(num_batches):
            # We decode the inputs we generated above because summarize() will tokenize them again properly 
            batch = tokenizer.decode(whole_inputs['input_ids'][0][i * token_batch_size : (i + 1) * token_batch_size])
            text_batches.append(batch)

        # Generate summaries for each batch
        batch_summaries = summarize(text_batches)
        text = ' '.join(batch_summaries)

    final_summary = ' '.join(summarize(text)[0].split())

    if return_partial_summaries:
        return final_summary, batch_summaries
    return final_summary


def summarize_video(url: str, token_batch_size: int = 1024):
    # Get video metadata
    metadata = get_video_metadata(url)
    title = metadata['title']

    # Get captions
    caption_file = get_caption_file_url(metadata, lang='en')
    if caption_file:
        auto_captions = requests.get(caption_file)
        assert auto_captions.ok, auto_captions.status_code
        text = get_text(auto_captions.json())
    else:
        pass  # TODO translate

    summary = summarize_long_text(text, token_batch_size=token_batch_size)
    return title, summary
