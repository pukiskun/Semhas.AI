import os
import pypdf
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, api_key: str = None):
        """
        Initializes the RAG Engine. Uses local offline SentenceTransformer.
        """
        # Load local embedding model from Hugging Face (downloads once, runs offline)
        # all-MiniLM-L6-v2 is extremely lightweight (~90MB) and fast on CPU
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def process_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[str, int]:
        """
        Parses the PDF, chunks the text, generates embeddings locally, and stores them in memory.
        Returns the extracted abstract (or a preview of the start of the paper) and the number of chunks.
        """
        # 1. Extract Text
        raw_text = self._extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            raise ValueError("No readable text found in the uploaded PDF. It might be scanned or empty.")
        
        # 2. Extract Abstract / Introduction preview for UI display
        abstract = self._extract_abstract(raw_text)

        # 3. Create Chunks
        self.chunks = self._chunk_text(raw_text, chunk_size, chunk_overlap)
        
        # 4. Generate Embeddings locally
        self._generate_embeddings()
        
        return abstract, len(self.chunks)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Helper to read text page-by-page from the PDF."""
        reader = pypdf.PdfReader(pdf_path)
        text_list = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_list.append(text)
        return "\n\n".join(text_list)

    def _extract_abstract(self, text: str) -> str:
        """Tries to find the abstract or returns the first 1200 characters of the paper."""
        lower_text = text.lower()
        abstract_start = lower_text.find("abstract")
        
        if abstract_start != -1:
            # Look for typical start of Introduction to cap the abstract
            intro_start = lower_text.find("introduction", abstract_start + 8)
            if intro_start != -1 and (intro_start - abstract_start) < 4000:
                return text[abstract_start:intro_start].strip()
            else:
                return text[abstract_start:abstract_start + 1500].strip() + "\n... [truncated]"
        
        return text[:1200].strip() + "\n... [truncated]"

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Chunks the text into overlapping segments, attempting to split at paragraphs 
        or sentences to maintain semantic boundaries.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If a single paragraph is larger than the chunk size, we split it by sentences
            if len(para) > chunk_size:
                sentences = para.replace(". ", ".\n").split("\n")
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if current_size + len(sent) > chunk_size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        # Handle overlap by taking the last part of the current chunk
                        overlap_words = " ".join(current_chunk).split()[-30:] if current_chunk else []
                        current_chunk = overlap_words + [sent]
                        current_size = sum(len(w) for w in current_chunk)
                    else:
                        current_chunk.append(sent)
                        current_size += len(sent)
            else:
                if current_size + len(para) > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    overlap_words = " ".join(current_chunk).split()[-30:] if current_chunk else []
                    current_chunk = overlap_words + [para]
                    current_size = sum(len(w) for w in current_chunk)
                else:
                    current_chunk.append(para)
                    current_size += len(para)
                    
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _generate_embeddings(self):
        """Generates embeddings for all stored chunks locally using Hugging Face SentenceTransformer."""
        if not self.chunks:
            return
        
        # encode chunks locally and normalize them so cosine similarity is just the dot product
        self.embeddings = self.model.encode(
            self.chunks, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves the top_k chunks most relevant to the query.
        Uses cosine similarity (via dot product since embeddings are normalized).
        """
        if len(self.chunks) == 0 or self.embeddings.size == 0:
            return []
            
        # Generate query embedding locally and normalize it
        query_emb = self.model.encode(
            query, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        # Calculate similarity (dot product of normalized vectors)
        similarities = np.dot(self.embeddings, query_emb)
        
        # Sort indices in descending order
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "content": self.chunks[idx],
                "similarity": float(similarities[idx])
            })
            
        return results
