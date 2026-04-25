from supabase import create_client, Client
from configuration.settings import settings
from pydantic import BaseModel


supabase_url = settings.SUPABASE_URL
supabase_key = settings.SUPABASE_KEY

supabase: Client = create_client(supabase_url=supabase_key,
                                 supabase_key=supabase_key)


class NewEmbeddingResponse(BaseModel):
    content: str
    embedding: list[float]


def insert_embedding(inputa_data: NewEmbeddingResponse):
    """
    Insert a new embedding into the database.
    """
    data = {
        "content": inputa_data.content,
        "embedding": inputa_data.embedding
    }
    response = supabase.table("embeddings").insert(data).execute()
    return response

