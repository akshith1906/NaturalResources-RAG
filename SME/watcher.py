import time
import subprocess
import os
import signal
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv 


load_dotenv()

DOCS_DIR = "./Docs"
GENERATED_FILES_DIR = "./generated_files"
SENT_FILES_MANIFEST = "sent_files_manifest.json" 
EMAIL_RECIPIENT = "vsai2k@gmail.com" 

APP_COMMAND = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
INGEST_COMMAND = [sys.executable, "ingest.py"]

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.example.com") 
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_SENDER_EMAIL = os.environ.get("SMTP_SENDER_EMAIL", "rag_agent@example.com")
SMTP_SENDER_PASSWORD = os.environ.get("SMTP_SENDER_PASSWORD", "mock_password")

if SMTP_SERVER == "smtp.example.com":
    print("\n\n!! WARNING: SMTP credentials still using mock defaults. Check .env file !!\n")
else:
    print(f"\n[DEBUG] Mailer initialized for server: {SMTP_SERVER} from {SMTP_SENDER_EMAIL}\n")


def _send_generated_file_email(file_path: Path, recipient_email: str) -> bool:
    if not all([SMTP_SERVER, SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD]) or SMTP_SERVER == "smtp.example.com":
        print("[Mailer] SMTP credentials not fully configured (using mock defaults). Skipping email.")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = f"RAG Agent: New File Generated - {file_path.name}"
        
        body = f"The RAG agent generated a new file for you: {file_path.name}"
        msg.attach(MIMEText(body, 'plain'))
        
        with open(file_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {file_path.name}")
        msg.attach(part)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD)
            text = msg.as_string()
            server.sendmail(SMTP_SENDER_EMAIL, recipient_email, text)
            
        print(f"[Mailer] Successfully sent {file_path.name} to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"[Mailer] Failed to send email for {file_path.name}: {e}")
        return False

def _load_sent_manifest() -> set:
    try:
        with open(SENT_FILES_MANIFEST, "r") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def _save_sent_manifest(sent_files: set):
    try:
        with open(SENT_FILES_MANIFEST, "w") as f:
            json.dump(list(sent_files), f, indent=2)
    except Exception as e:
        print(f"[Mailer] Failed to save manifest: {e}")


class IngestionHandler(FileSystemEventHandler):
    def __init__(self, process_manager):
        self.process_manager = process_manager
        self.last_triggered = 0
        self.debounce_period = 5 

    def on_any_event(self, event):
        if event.is_directory:
            return
        
        current_time = time.time()
        if (current_time - self.last_triggered) < self.debounce_period:
            print(f"[Watcher] Change detected in {DOCS_DIR}, but waiting for debounce period to end...")
            return

        print(f"\n--- [Watcher] Ingestion change detected: {event.src_path} ---")
        self.last_triggered = current_time
        self.process_manager.trigger_reload()


class GeneratedFilesHandler(FileSystemEventHandler):
    def __init__(self):
        self.sent_files = _load_sent_manifest()
        self.debounce_times = {}
        self.debounce_period = 2 

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        file_name = file_path.name
        
        current_time = time.time()
        if file_name in self.debounce_times and (current_time - self.debounce_times[file_name] < self.debounce_period):
            return
        
        self.debounce_times[file_name] = current_time

        if file_name in self.sent_files:
            return
        
        time.sleep(0.5) 

        print(f"[Mailer] New file created: {file_name}. Preparing email...")
        
        if _send_generated_file_email(file_path, EMAIL_RECIPIENT):
            self.sent_files.add(file_name)
            _save_sent_manifest(self.sent_files)
        else:
            print(f"[Mailer] Failed to send email for {file_path.name}.")


class ProcessManager:
    def __init__(self):
        self.server_process = None
        self.is_reloading = False

    def start_server(self):
        print("[Watcher] Starting FastAPI server...")
        self.server_process = subprocess.Popen(APP_COMMAND)
        print(f"[Watcher] Server running with PID: {self.server_process.pid}")

    def stop_server(self):
        if self.server_process:
            print(f"[Watcher] Stopping FastAPI server (PID: {self.server_process.pid})...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
                print("[Watcher] Server stopped.")
            except subprocess.TimeoutExpired:
                print("[Watcher] Server did not stop, killing...")
                self.server_process.kill()
                print("[Watcher] Server killed.")
            self.server_process = None

    def trigger_reload(self):
        if self.is_reloading:
            print("[Watcher] Reload already in progress, skipping.")
            return

        self.is_reloading = True
        print("--- [Watcher] RELOAD TRIGGERED ---")
        self.stop_server()
        
        print("[Watcher] Running ingest.py...")
        try:
            subprocess.run(INGEST_COMMAND, check=True)
            print("[Watcher] Ingestion complete.")
        except subprocess.CalledProcessError as e:
            print(f"[Watcher] ERROR: ingest.py failed: {e}")
        except Exception as e:
            print(f"[Watcher] An unexpected error occurred during ingestion: {e}")
        
        self.start_server()
        
        print("--- [Watcher] RELOAD COMPLETE ---")
        time.sleep(2)
        self.is_reloading = False

if __name__ == "__main__":
    print("--- Starting File Watcher and Server Manager ---")
    
    Path(DOCS_DIR).mkdir(exist_ok=True)
    Path(GENERATED_FILES_DIR).mkdir(exist_ok=True)
    
    if not os.path.exists(DOCS_DIR):
        print(f"Error: '{DOCS_DIR}' directory not found.")
        sys.exit(1)

    manager = ProcessManager()
    manager.start_server()

    observer = Observer()
    
    ingestion_handler = IngestionHandler(manager)
    observer.schedule(ingestion_handler, DOCS_DIR, recursive=True)
    print(f"[Watcher] Watching for ingestion changes in: {os.path.abspath(DOCS_DIR)}")

    email_handler = GeneratedFilesHandler()
    observer.schedule(email_handler, GENERATED_FILES_DIR, recursive=False) 
    print(f"[Watcher] Watching for new generated files in: {os.path.abspath(GENERATED_FILES_DIR)}")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Watcher] Shutdown signal received...")
        observer.stop()
        manager.stop_server()
    
    observer.join()
    print("[Watcher] Exited gracefully.")