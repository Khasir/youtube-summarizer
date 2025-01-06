import json
from flask import Flask, request
from app.youtube_summarizer import summarize_video


app = Flask(__name__)


@app.route("/", methods=["GET"])
def main():
    return "", 200

@app.route("/summarize", methods=["GET"])
def summarize():
    video_url = request.args.get('url')
    if not video_url:
        return "", 200
    
    title, summary = summarize_video(video_url, token_batch_size=100)
    return json.dumps({
        "title": title,
        "summary": summary,
        "url": video_url
    })


@app.get("/robots.txt")
def robots():
    """
    Disallow all robots.
    """
    return "User-agent: *\nDisallow: /", 200


if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True)
