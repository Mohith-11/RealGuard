from fastapi import FastAPI

from app.schemas import Transaction
from app.predictor import predict

app = FastAPI(

    title="RealGuard Fraud Detection",

    version="1.0"

)


@app.get("/")
def root():

    return {

        "message": "RealGuard API Running"

    }


@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


@app.post("/predict")
def fraud_predict(transaction: Transaction):

    result = predict(

        transaction.model_dump()

    )

    return result

