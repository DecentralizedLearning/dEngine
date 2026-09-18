from pydantic import BaseModel


class ClientListSchema(BaseModel):
    UUID: str
