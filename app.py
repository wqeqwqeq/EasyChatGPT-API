from flask import Flask, render_template, request
from markupsafe import Markup
import openai
import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
import base64

import os
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time, sleep
from uuid import uuid4
import datetime


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


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime(
        "%A, %B %d, %Y at %I:%M%p %Z"
    )


def gpt3_embedding(content, engine="text-embedding-ada-002"):
    content = content.encode(encoding="ASCII", errors="ignore").decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response["data"][0]["embedding"]  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i["vector"]:
            # skip this one because it is the same message
            continue
        score = similarity(i["vector"], vector)
        i["score"] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d["score"], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo():
    files = os.listdir("chat_logs")
    files = [i for i in files if ".json" in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json("chat_logs/%s" % file)
        result.append(data)
    ordered = sorted(
        result, key=lambda d: d["time"], reverse=False
    )  # sort them all chronologically
    return result


def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(
        memories, key=lambda d: d["time"], reverse=False
    )  # sort them chronologically
    block = ""
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem["message"] + "\n\n"
        identifiers.append(mem["uuid"])
        timestamps.append(mem["time"])
    block = block.strip()
    prompt = open_file("prompt_notes.txt").replace("<<INPUT>>", block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    ####   SAVE NOTES
    vector = gpt3_embedding(block)
    info = {
        "notes": notes,
        "uuids": identifiers,
        "times": timestamps,
        "uuid": str(uuid4()),
        "vector": vector,
    }
    filename = "notes_%s.json" % time()
    save_json("notes/%s" % filename, info)
    return notes


def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ""
    for i in short:
        output += "%s\n\n" % i["message"]
    output = output.strip()
    return output


def gpt3_completion(
    prompt,
    engine="text-davinci-003",
    temp=0.0,
    top_p=1.0,
    tokens=400,
    freq_pen=0.0,
    pres_pen=0.0,
    stop=["USER:", "RAVEN:"],
):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding="ASCII", errors="ignore").decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop,
            )
            text = response["choices"][0]["text"].strip()
            text = re.sub("[\r\n]+", "\n", text)
            text = re.sub("[\t ]+", " ", text)
            filename = "%s_gpt3.txt" % time()
            if not os.path.exists("gpt3_logs"):
                os.makedirs("gpt3_logs")
            save_file("gpt3_logs/%s" % filename, prompt + "\n\n==========\n\n" + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print("Error communicating with OpenAI:", oops)
            sleep(1)


def rm_files_in_dir(folder):
    dirs = os.listdir(folder)
    for f in dirs:
        path = os.path.join(folder, f)
        os.remove(path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.form["user_input"]

    timestamp = time()
    vector = gpt3_embedding(user_input)
    timestring = timestamp_to_datetime(timestamp)
    message = "%s\n%s: %s" % (timestring, "User", user_input)
    info = {
        "speaker": "USER",
        "time": timestamp,
        "vector": vector,
        "message": message,
        "uuid": str(uuid4()),
        "timestring": timestring,
    }
    filename = "log_%s_USER.json" % timestamp
    save_json("chat_logs/%s" % filename, info)
    #### load conversation
    conversation = load_convo()
    #### compose corpus (fetch memories, etc)
    memories = fetch_memories(vector, conversation, 10)  # pull episodic memories
    # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
    notes = summarize_memories(memories)
    # TODO - search existing notes first
    recent = get_last_messages(conversation, 4)
    prompt = (
        open_file("prompt_response.txt")
        .replace("<<NOTES>>", notes)
        .replace("<<CONVERSATION>>", recent)
    )
    #### generate response, vectorize, save, etc
    output = gpt3_completion(prompt)
    timestamp = time()
    vector = gpt3_embedding(output)
    timestring = timestamp_to_datetime(timestamp)
    message = "%s\n%s: %s" % (timestring, "RAVEN", output)
    info = {
        "speaker": "RAVEN",
        "time": timestamp,
        "vector": vector,
        "message": message,
        "uuid": str(uuid4()),
        "timestring": timestring,
    }
    filename = "log_%s_RAVEN.json" % time()
    save_json("chat_logs/%s" % filename, info)
    #### print output
    # print("\n\nRAVEN: %s" % output)

    # print(user_input)
    # messages.append({"role": "user", "content": user_input})
    # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    # ai_response = completion.choices[0].message["content"]
    # print(ai_response)
    # messages.append({"role": "assistant", "content": ai_response})
    # print(messages)
    print(prompt)
    return Markup(markdown.markdown(output, extensions=["fenced_code", "codehilite"]))


@app.route("/reset")
def reset():
    global messages
    messages = []
    rm_files_in_dir("chat_logs")
    rm_files_in_dir("gpt3_logs")
    rm_files_in_dir("notes")

    return "Conversation history has been reset."


if __name__ == "__main__":
    app.run(debug=True)
