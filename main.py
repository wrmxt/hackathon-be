from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

from api import api  # noqa: E402

origins = [
    "http://localhost:5173",   # Vite dev server
    # "http://127.0.0.1:5173", # add this too if you sometimes use 127.0.0.1
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # or ["*"] for all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],           # allow all HTTP methods
    allow_headers=["*"],           # allow all headers
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
