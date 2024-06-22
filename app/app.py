from flask import Flask


app = Flask(__name__)


@app.route("/", methods=["GET"])
def main():
    return "", 200


@app.get("/robots.txt")
def robots():
    """
    Disallow all robots.
    """
    return "User-agent: *\nDisallow: /", 200


if __name__ == "__main__":
    app.run(host='localhost', port=8080, debug=True)
