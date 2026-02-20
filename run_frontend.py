import webbrowser
import subprocess
import time

# Start the local server
process = subprocess.Popen(["python", "-m", "http.server", "5500"])

# Wait for server to start
time.sleep(2)

# Open frontend in browser automatically
url = "http://127.0.0.1:5500/index.html"
print(f"🌐 Opening frontend: {url}")
webbrowser.open(url)

# Keep server alive
process.wait()