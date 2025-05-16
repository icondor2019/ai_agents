from sentence_transformers import SentenceTransformer
# from loguru import logger


def embedding_text(text_list: list) -> list:
    # model = SentenceTransformer('intfloat/e5-base-v2')
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)
    input_texts = text_list
    embeddings = model.encode(input_texts, normalize_embeddings=True)
    return embeddings
