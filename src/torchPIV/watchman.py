import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class EventPrinter(FileSystemEventHandler):
    def on_created(self, event):
        print(event.src_path, flush=True)
        pass

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = EventPrinter()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()