import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.inference import correct_text

api_router = APIRouter()


class ResponseModel(BaseModel):
    corrected: str
    delay: float = Field(
        ...,
        description="Time taken to process the request in seconds",
        alias="delay_in_seconds",
    )


class RequestModel(BaseModel):
    text: str


@api_router.post("/correct")
async def correct(body: RequestModel):
    start = time.time()
    corrected = correct_text(body.text)
    end = time.time()
    return ResponseModel(corrected=corrected, delay_in_seconds=end - start)
