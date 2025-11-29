from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    message: str

class ReturnBorrowingRequest(BaseModel):
    borrowing_id: str


class ConfirmBorrowingRequest(BaseModel):
    borrowing_id: str
    owner_id: str
