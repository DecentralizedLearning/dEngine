from pydantic import BaseModel


class ClientListSchema(BaseModel):
    UUID: str


class PresignedModelDownloadURL(BaseModel):
    url: str


class JWTTokenPresignedModelDownloadURL(BaseModel):
    UUID: str
    token_created_at: str
