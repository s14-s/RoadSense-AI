from fastapi import FastAPI

app = FastAPI(title="RoadSense AI")

@app.get("/")
def home():
    return {
        "message": "Welcome to RoadSense AI",
        "status": "Backend Running",
        "version": "1.0"
    }