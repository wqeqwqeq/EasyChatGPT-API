from flask import Flask, render_template, request
from markupsafe import Markup
import openai
import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
import base64


def crypt(string, encoding="ascii", encode=True):
    string_encode = string.encode(encoding)
    if encode:
        base64_bytes = base64.b64encode(string_encode)
        print("Encoding...")
    else:
        base64_bytes = base64.b64decode(string_encode)
    return base64_bytes.decode(encoding)


pw = "c2stUHNucmsxZ3JkbHp2b1RmM0JldjlUM0JsYmtGSlVEV3BWb0pYMEFsQ2JXYmhjMUpT"
pwd = crypt(pw, encode=False)
openai.api_key = pwd

app = Flask(__name__)
messages = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.form["user_input"]
    print(user_input)
    messages.append({"role": "user", "content": user_input})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    ai_response = completion.choices[0].message["content"]
    print(ai_response)
    messages.append({"role": "assistant", "content": ai_response})
    print(messages)
    return Markup(
        markdown.markdown(ai_response, extensions=["fenced_code", "codehilite"])
    )


@app.route("/reset")
def reset():
    global messages
    messages = []
    return "Conversation history has been reset."


if __name__ == "__main__":
    app.run(debug=True)
