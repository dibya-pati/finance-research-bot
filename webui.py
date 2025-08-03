import os
import uuid
from datetime import datetime
from flask import (
    Flask,
    render_template_string,
    request,
    send_from_directory,
    session,
    redirect,
    url_for,
)
from research_agent import (
    create_research_agent_graph,
    extract_and_save_dual_output,
    get_prior_context,
)

app = Flask(__name__)
app.secret_key = os.environ.get("WEBUI_SECRET_KEY", str(uuid.uuid4()))

# Minimal HTML template with chat and download
TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Research Agent Chat</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 0; }
        .container { max-width: 700px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 24px; }
        h1 { text-align: center; }
        .chat-box { height: 350px; overflow-y: auto; border: 1px solid #ddd; padding: 12px; background: #fafafa; margin-bottom: 16px; }
        .msg { margin-bottom: 12px; }
        .user { color: #2c3e50; font-weight: bold; }
        .bot { color: #2980b9; }
        .download-link { margin-top: 16px; display: block; text-align: right; }
        .input-row { display: flex; gap: 8px; }
        input[type=text] { flex: 1; padding: 8px; border-radius: 4px; border: 1px solid #ccc; }
        button { padding: 8px 18px; border-radius: 4px; border: none; background: #2980b9; color: #fff; font-weight: bold; cursor: pointer; }
        button:disabled { background: #b0b0b0; cursor: not-allowed; }
        button:hover:enabled { background: #1c5d8c; }
        .spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid #f3f3f3; border-top: 3px solid #2980b9; border-radius: 50%; animation: spin 1s linear infinite; margin-left: 10px; vertical-align: middle; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Research Agent Chat</h1>
        <div class=\"chat-box\" id=\"chat-box\">
            {% for m in messages %}
                <div class=\"msg\"><span class=\"user\">You:</span> {{ m['user'] }}</div>
                <div class=\"msg\"><span class=\"bot\">Bot:</span> {{ m['bot']|safe }}</div>
                {% if m['filename'] %}
                    <a class=\"download-link\" href=\"/download/{{ m['filename']|replace('reports/', '') }}\" target=\"_blank\">Download HTML Report</a>
                {% endif %}
            {% endfor %}
        </div>
        <form method=\"post\" class=\"input-row\" id=\"chat-form\" autocomplete=\"off\">
            <input type=\"text\" name=\"user_input\" placeholder=\"Type your question...\" autofocus required id=\"user-input\">
            <button type=\"submit\" id=\"send-btn\">Send</button>
            <div id=\"spinner\" class=\"spinner\" style=\"display:none;\"></div>
        </form>
    </div>
    <script>
        const form = document.getElementById('chat-form');
        const sendBtn = document.getElementById('send-btn');
        const spinner = document.getElementById('spinner');
        const userInput = document.getElementById('user-input');
        form.addEventListener('submit', function() {
            sendBtn.disabled = true;
            spinner.style.display = 'inline-block';
        });
        window.onload = function() {
            sendBtn.disabled = false;
            spinner.style.display = 'none';
            userInput.focus();
        };
    </script>
</body>
</html>
"""

# Ensure reports directory exists
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)

# Setup agent
agent_app = create_research_agent_graph()


@app.route("/", methods=["GET", "POST"])
def chat():
    if "thread_id" not in session:
        session["thread_id"] = (
            f"webui_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"
        )
    if "user_id" not in session:
        session["user_id"] = "webui_user"
    if "messages" not in session:
        session["messages"] = []

    messages = session["messages"]
    filename = None
    if request.method == "POST":
        user_input = request.form["user_input"].strip()
        config = {
            "configurable": {
                "thread_id": session["thread_id"],
                "user_id": session["user_id"],
            }
        }
        from langchain_core.messages import HumanMessage

        prior_messages_with_system = get_prior_context(config)
        user_message = HumanMessage(content=user_input)
        all_messages = prior_messages_with_system + [user_message]
        result = agent_app.invoke(
            {
                "messages": all_messages,
                "thread_id": session["thread_id"],
                "user_id": session["user_id"],
            },
            config=config,
        )
        final_message = result["messages"][-1]
        sections = extract_and_save_dual_output(final_message.content)
        filename = sections["filename"]
        messages.append(
            {"user": user_input, "bot": sections["console"], "filename": filename}
        )
        session["messages"] = messages
        return redirect(url_for("chat"))
    return render_template_string(TEMPLATE, messages=messages)


@app.route("/download/<path:filename>")
def download(filename):
    # Remove any leading directory for security and compatibility
    safe_filename = os.path.basename(filename)
    return send_from_directory(REPORTS_DIR, safe_filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=7860)
