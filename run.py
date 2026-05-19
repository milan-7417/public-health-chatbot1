import multiprocessing
import webbrowser
import time

def start_server():
    import uvicorn
    from backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Start backend
    p = multiprocessing.Process(target=start_server)
    p.daemon = True
    p.start()

    # Wait for backend to start
    time.sleep(5)

    # Open browser
    webbrowser.open("http://127.0.0.1:8000")

    # Keep app alive
    p.join()