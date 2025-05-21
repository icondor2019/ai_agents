from sentence_transformers import SentenceTransformer
from openai import OpenAI
from configuration.settings import settings
from loguru import logger

class EmbeddingTools():
    def __init__(self):
        openai_api_key = settings.OPENAI_API_KEY
        self.client = OpenAI(openai_api_key=openai_api_key)
        
    def embedding_text(self, text_list: list) -> list:
        # model = SentenceTransformer('intfloat/e5-base-v2')
        model_name = 'intfloat/e5-base-v2'
        # model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        model = SentenceTransformer(model_name)
        input_texts = text_list
        embeddings = model.encode(input_texts, normalize_embeddings=True)
        return embeddings

    def get_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding
