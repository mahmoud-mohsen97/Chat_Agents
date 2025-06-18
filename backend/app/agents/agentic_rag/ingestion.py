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


def ingest_pdf(pdf_path: str, 
               collection="pdf_pages",
               host="localhost",
               port=6333):
    """Ingest PDF as images into Qdrant vector store."""
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


class QdrantRetriever:
    """Simple Qdrant retriever for compatibility with existing code."""
    
    def __init__(self, collection="pdf_pages", host="localhost", port=6333, k=1):
        self.client = QdrantClient(host=host, port=port)
        self.embeddings = CohereMultimodalEmbeddings()
        self.collection = collection
        self.k = k
    
    def invoke(self, query: str):
        """Search for similar documents and return in LangChain Document format."""
        # Create query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Search
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=self.k
        )
        
        # Convert to Document format for compatibility
        from langchain_core.documents import Document
        docs = []
        for result in results:
            doc = Document(
                page_content=result.payload["page_content"],
                metadata={
                    "page": result.payload["page"],
                    "source": result.payload.get("source", ""),
                    "score": result.score
                }
            )
            docs.append(doc)
        
        return docs


def get_retriever(collection="pdf_pages", host="localhost", port=6333):
    """Get retriever from existing Qdrant collection."""
    return QdrantRetriever(collection=collection, host=host, port=port)


# Initialize the retriever (will work after ingestion)
retriever = None


if __name__ == "__main__":
    # Ingest the PDF
    pdf_path = "Mahmoud-Mohsen-CV.pdf"
    client, collection = ingest_pdf(pdf_path)
    print("PDF ingested successfully!")
    
    # Initialize retriever after ingestion
    retriever = get_retriever()
    
    # Test retrieval
    test_query = "What are the required skills?"
    results = retriever.invoke(test_query)
    print(f"Retrieved {len(results)} documents for query: '{test_query}'")
    for i, doc in enumerate(results):
        print(f"Document {i+1}: Page {doc.metadata['page']}, Score: {doc.metadata.get('score', 'N/A')}")
