from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    user_id: str
    message: str

class ReturnBorrowingRequest(BaseModel):
    borrowing_id: str


class ConfirmBorrowingRequest(BaseModel):
    borrowing_id: str
    owner_id: str

# Новый модель для обновления предмета
class UpdateItemRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None

# Запрос на создание (request) займа вещи
class RequestBorrowingRequest(BaseModel):
    user_id: str  # borrower
    item_id: str
