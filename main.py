from fastapi import FastAPI
import uvicorn
app = FastAPI()

from api import api  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
