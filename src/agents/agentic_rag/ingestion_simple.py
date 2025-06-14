import os
import base64
import io
import warnings
from typing import List
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


def pdf_to_images(path: str, zoom: float = 1.5) -> List[Image.Image]:
    """Convert PDF pages to images using PyMuPDF."""
    doc = fitz.open(path)
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def image_to_data_url(img: Image.Image, fmt="PNG") -> str:
    """Convert PIL Image to base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


class CohereMultimodalEmbeddings:
    """Custom Cohere multimodal embeddings class."""
    
    def __init__(self, model="embed-v4.0"):
        self.client = cohere.ClientV2(os.environ["COHERE_API_KEY"])
        self.model = model

    def embed_documents(self, imgs):
        """Embed image documents (data-URL strings)."""
        inputs = [{"content": [{"type": "image_url", 
                               "image_url": {"url": url}}]} for url in imgs]
        res = self.client.embed(
            model=self.model,
            inputs=inputs,
            input_type="search_document",
            embedding_types=["float"]
        )
        return res.embeddings.float

    def embed_query(self, query):
        """Embed text query."""
        res = self.client.embed(
            model=self.model,
            texts=[query],
            input_type="search_query",
            embedding_types=["float"]
        )
        return res.embeddings.float[0]


def ingest_pdf_simple(pdf_path: str, 
                      collection="pdf_pages",
                      host="localhost",
                      port=6333):
    """Ingest PDF as images into Qdrant vector store using direct client."""
    print(f"Processing PDF: {pdf_path}")
    
    # Convert PDF to images
    imgs = pdf_to_images(pdf_path)
    print(f"Converted PDF to {len(imgs)} images")
    
    # Convert images to data URLs
    data_urls = [image_to_data_url(img) for img in imgs]
    print(f"Converted {len(data_urls)} images to base64 data URLs")
    
    # Initialize embeddings and client
    embeddings = CohereMultimodalEmbeddings()
    client = QdrantClient(host=host, port=port)
    
    # Create embeddings
    print("Creating embeddings...")
    vectors = embeddings.embed_documents(data_urls)
    print(f"Created {len(vectors)} embeddings")
    
    # Create collection if it doesn't exist
    try:
        collection_info = client.get_collection(collection)
        print(f"Collection '{collection}' already exists")
    except:
        print(f"Creating collection '{collection}'")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=len(vectors[0]),
                distance=Distance.COSINE
            )
        )
    
    # Prepare points for insertion
    points = []
    for i, (vector, data_url) in enumerate(zip(vectors, data_urls)):
        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "page_content": data_url,
                "page": i,
                "source": pdf_path
            }
        )
        points.append(point)
    
    # Insert points
    print("Inserting points into Qdrant...")
    client.upsert(collection_name=collection, points=points)
    print(f"Successfully inserted {len(points)} points into collection '{collection}'")
    
    return client, collection


def search_documents(query: str, 
                     collection="pdf_pages",
                     host="localhost",
                     port=6333,
                     k=3):
    """Search for similar documents."""
    client = QdrantClient(host=host, port=port)
    embeddings = CohereMultimodalEmbeddings()
    
    # Create query embedding
    query_vector = embeddings.embed_query(query)
    
    # Search
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=k
    )
    
    return results


if __name__ == "__main__":
    # Ingest the PDF
    pdf_path = "Digital Innovation engineer JD.pdf"
    client, collection = ingest_pdf_simple(pdf_path)
    
    # Test search
    print("\nTesting search...")
    query = "What are the required skills?"
    results = search_documents(query)
    
    print(f"Found {len(results)} results for query: '{query}'")
    for i, result in enumerate(results):
        print(f"Result {i+1}: Score={result.score:.4f}, Page={result.payload['page']}")
    
    print("\nIngestion completed successfully!") 