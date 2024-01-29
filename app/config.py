config = {
    'milvus': {
        'host': 'standalone',
        'port': '19530',
        'collection_name': 'wiki_embeddings'
    },
    'wiki_titles': {
        'Boiler',
        'Snorkeling',
        'Minimum_viable_product',
        'Juniper',
        'York'
    },
    'input_file': '/app/data/articles.json',
    'model_dimension': 768,
    'model_name': 'bert-base-uncased',
}
