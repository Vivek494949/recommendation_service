from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import os
from typing import List
import uvicorn

class EmbeddingRequest(BaseModel):
    book_id: str
    title: str
    description: str
    text: str
    author_address: str

app = FastAPI(title="Book Recommendation Service")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database connection
def get_db_connection():
    return psycopg2.connect(os.getenv('DATABASE_URL'))

@app.on_event("startup")
def startup():
    """Initialize database connection on startup"""
    try:
        conn = get_db_connection()
        conn.close()
        print("✓ Database connected successfully")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/generate-embedding")
def generate_embedding(book_id: str = Query(...)):
    """Generate embedding for a book - this is what the Next.js API calls"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if embedding already exists
        cur.execute("SELECT book_id FROM book_embeddings WHERE book_id = %s", (book_id,))
        existing = cur.fetchone()
        
        if existing:
            # Return existing embedding data
            cur.execute("SELECT title, description, embedding, author_address FROM book_embeddings WHERE book_id = %s", (book_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return {
                "book_id": book_id,
                "title": row[0],
                "description": row[1],
                "embedding": row[2] if isinstance(row[2], list) else list(row[2]),
                "author": row[3]
            }
        
        # For new books, return placeholder - actual embedding stored via /store-embedding
        cur.close()
        conn.close()
        return {
            "book_id": book_id,
            "title": f"Book {book_id}",
            "description": f"Book description for {book_id}",
            "embedding": [0.0] * 384,  # Placeholder embedding
            "author": "0x0000000000000000000000000000000000000000"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store-embedding")
def store_embedding(request: EmbeddingRequest):
    """Store book embedding in database"""
    try:
        embedding = model.encode(request.text, convert_to_tensor=False).tolist()
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert or update embedding
        cur.execute("""
            INSERT INTO book_embeddings (book_id, title, description, embedding, author_address)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (book_id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                updated_at = CURRENT_TIMESTAMP
        """, (request.book_id, request.title, request.description, embedding, request.author_address))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {
            "status": "success",
            "message": "Embedding stored",
            "book_id": request.book_id,
            "embedding": embedding
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend")
def recommend(book_id: str = Query(...), limit: int = Query(5, ge=1, le=20)):
    """Get book recommendations based on similarity"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get the query book
        cur.execute("SELECT embedding, title FROM book_embeddings WHERE book_id = %s", (book_id,))
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Book not found")
        
        query_embedding = result[0]
        
        # Get all books
        cur.execute("""
            SELECT book_id, title, description, author_address, 
                   embedding <-> %s AS distance
            FROM book_embeddings
            WHERE book_id != %s
            ORDER BY distance
            LIMIT %s
        """, (query_embedding, book_id, limit))
        
        recommendations = []
        for row in cur.fetchall():
            recommendations.append({
                "book_id": row[0],
                "title": row[1],
                "description": row[2],
                "author": row[3],
                "similarity_score": float(1 / (1 + row[4]))  # Convert distance to similarity
            })
        
        cur.close()
        conn.close()
        
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
