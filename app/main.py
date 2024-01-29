import json
import click
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

from config import config
from tools.process_articles import download_wikipedia_articles

# Use a pre-trained transformer model for vectorization
model_name = config.get('model_name', 'bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)


def vectorize_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def connect_and_create_collection(collection_name, dimension):
    connections.connect(
        host=config['milvus']['host'],
        port=config['milvus']['port'])

    if utility.has_collection(collection_name):
        Collection(collection_name).drop()

    field_embedding = FieldSchema(
        name='embedding',
        dtype=DataType.FLOAT_VECTOR,
        dim=dimension)
    field_id = FieldSchema(name='id', dtype=DataType.INT64, is_primary=True)
    schema = CollectionSchema(
        fields=[
            field_id,
            field_embedding],
        description="Wikipedia Embeddings Collection")
    Collection(collection_name, schema)


def insert_vectors(collection_name, vectors):
    collection = Collection(collection_name)
    schema = collection.schema

    if not vectors.any():
        print("No vectors to insert.")
        return

    # Get the field corresponding to 'embedding' in the schema
    embedding_field = next(
        field for field in schema.fields if field.name == 'embedding')

    # Ensure all vectors have the correct dimension
    vector_dim = embedding_field.params['dim']

    if len(vectors.shape) == 1:
        # Handle 1-dimensional tensor differently
        vectors = vectors.unsqueeze(0)

    if vectors.shape[1] != vector_dim:
        print("Invalid vector dimension. Expected:",
              vector_dim, "Got:", vectors.shape[1])
        return

    # Generate IDs for vectors
    ids = torch.arange(vectors.shape[0], dtype=torch.int64)

    # Prepare data for insertion
    data_to_insert = [{'id': i.item(), 'embedding': vec.tolist()}
                      for i, vec in zip(ids, vectors)]

    # Insert data into the collection
    collection.insert(data_to_insert)
    print(f"Inserted {len(data_to_insert)} vectors into the collection.")


def create_index(collection_name):
    collection = Collection(collection_name)
    try:
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {
                "nlist": 1
            }
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params,
            index_name="article"
        )

        print("Index created successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")


def search_vectors(collection_name, query_text, top_k=5):
    query_vector = vectorize_text(query_text)
    collection = Collection(collection_name)

    collection.load()

    search_param = {
        'metric_type': 'L2',
        'params': {'nprobe': 1}
    }

    # Ensure query_vector is a list of floats
    query_data = [float(value) for value in query_vector]

    # Use a list to store the query data, as Milvus search method expects a
    # list
    results = collection.search(
        data=[query_data],
        anns_field='embedding',
        param=search_param,
        limit=top_k)
    return results


@click.command()
@click.option('--user-query', '-q', default='some search query',
              help='User search query')
@click.option('--download', '-d', is_flag=True, help='Download wiki articles')
def main(user_query, download):
    if download:
        download_wikipedia_articles(config['wiki_titles'])
        return

    with open(config['input_file'], 'r', encoding='utf-8') as file:
        articles_data = json.load(file)

    # Assuming 'text' key contains the list of strings
    articles = articles_data.get('text', [])

    collection_name = config['milvus']['collection_name']
    # Dimension of BERT-based embeddings
    dimension = config.get('model_dimension', 768)

    connect_and_create_collection(collection_name, dimension)

    all_embeddings = torch.stack([vectorize_text(text) for text in articles])
    insert_vectors(collection_name, all_embeddings)

    create_index(collection_name)

    # Check if the vector length is correct
    results = search_vectors(collection_name, user_query)
    print(results)


if __name__ == "__main__":
    main()
