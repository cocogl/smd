from flask import Flask, render_template, redirect, url_for
import subprocess
import sys
import os
from pathlib import Path

# Flask 앱이 templates 폴더 안에 있으므로, 현재 폴더를 템플릿 폴더로 사용하도록 지정
app = Flask(__name__, template_folder=str(Path(__file__).parent))

# Keep reference if we want to avoid multiple launches
launched_procs = {}


def launch_projection_mapper():
    proj_path = Path(__file__).parent / "projection_mapper.py"
    python_exec = sys.executable
    # Start non-blocking; create a new process group on Windows to keep it alive
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen([python_exec, str(proj_path)], creationflags=creationflags)
    launched_procs["class_a"] = proc
    return proc


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/launch/class-a", methods=["POST"]) 
def launch_class_a():
    launch_projection_mapper()
    # Redirect back to home with a simple message (optional: flash)
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Run Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
