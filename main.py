from io import BytesIO

import pandas as pd

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.modeling import run_modeling


app = FastAPI(
    title="John Barandica Data Analyzer API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://johnbarandica.com",
        "https://www.johnbarandica.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():

    return {
        "status": "online",
        "service":
            "John Barandica Data Analyzer API",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target: str = Form(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )


    if not file.filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are currently supported."
        )


    try:

        contents = await file.read()


        dataframe = pd.read_csv(
            BytesIO(contents)
        )


        results = run_modeling(
            dataframe,
            target
        )


        return results


    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "The dataset could not be processed: "
                + str(exc)
            )
        )
