import traceback
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from time import time

logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG level
logger = logging.getLogger(__name__)


@dataclass
class NostrEvent:
    id: str
    content: str
    pubkey: str
    created_at: int
    kind: int


def visualize_vector(vector: np.ndarray, width: int = 32) -> str:
    return "".join("██" if bit else "  " for bit in vector.flatten())


class NostrBinaryVectorDB:
    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-embed-xsmall-v1",
        dimension: int = 384,
    ):
        """
        Initialize the binary vector database.

        Args:
            model_name: Name of the sentence transformer model
            dimension: Original embedding dimension

        Note:
            For binary indices, FAISS expects the dimension in bytes (not bits).
            So if we have a 384-dimensional vector, we need ceil(384/8) = 48 bytes
            to store it in binary format.
        """
        self.model = SentenceTransformer(model_name)
        self.original_dimension = dimension
        # Convert dimension to number of bytes needed (8 bits per byte)
        self.binary_dimension = (
            (dimension + 7) // 8 * 8
        )  # Round up to nearest multiple of 8
        self.code_size = self.binary_dimension // 8  # Dimension in bytes for FAISS
        self.index = faiss.IndexBinaryFlat(self.binary_dimension)
        self.events: Dict[int, NostrEvent] = {}

    def fetch_events(self, limit: int, relays: List[str] = None) -> List[NostrEvent]:
        try:
            cmd = ["nak", "req", "-k", "1", "--limit", str(limit)]
            if relays:
                cmd.extend(relays)

            logger.debug(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stderr
                )

            events = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        event_data = json.loads(line)
                        event = NostrEvent(
                            id=event_data["id"],
                            content=event_data["content"],
                            pubkey=event_data["pubkey"],
                            created_at=event_data["created_at"],
                            kind=event_data["kind"],
                        )
                        events.append(event)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse line: {line}")
                        logger.error(f"JSON error: {str(e)}")
                        raise

            logger.info(f"Successfully fetched {len(events)} events")
            return events

        except Exception as e:
            logger.error(f"Error in fetch_events: {str(e)}")
            logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
            raise

    def add_events(self, events: List[NostrEvent]) -> None:
        if not events:
            return

        binary_embeddings = self.create_binary_embeddings(events)
        self.index.add(binary_embeddings)

        start_idx = len(self.events)
        for i, event in enumerate(events, start=start_idx):
            self.events[i] = event

        logger.info(f"Added {len(events)} events to the database")

    def create_binary_embeddings(self, events: List[NostrEvent]) -> np.ndarray:
        """
        Create binary embeddings from event contents.

        This process involves:
        1. Getting dense embeddings from the transformer model
        2. Normalizing them
        3. Converting to binary (0s and 1s) based on sign
        4. Packing into bytes for FAISS binary index
        """
        # Get dense embeddings
        contents = [event.content for event in events]
        dense_embeddings = self.model.encode(contents, convert_to_tensor=True)
        dense_embeddings = dense_embeddings.cpu().numpy()

        # Normalize the vectors
        faiss.normalize_L2(dense_embeddings)

        # Convert to binary (True/False) based on sign
        binary_values = dense_embeddings > 0

        # Pack into bytes (uint8)
        packed = np.packbits(binary_values, axis=1)

        # Ensure we have the correct number of bytes
        assert (
            packed.shape[1] == self.code_size
        ), f"Expected {self.code_size} bytes but got {packed.shape[1]}"

        return packed

    def search(
        self, query: str, k: int = 5, include_opposite: bool = True
    ) -> Dict[str, List[Tuple[NostrEvent, float]]]:
        """
        Search for similar events using Hamming distance in binary space.

        The process works like this:
        1. We convert the query text into a dense embedding
        2. We convert that embedding into a binary vector
        3. We search using Hamming distance (count of differing bits)
        4. We convert Hamming distances to similarity scores

        For binary vectors, a smaller Hamming distance means greater similarity.
        We convert the Hamming distance to a similarity score between 0 and 1
        where 1 means identical and 0 means completely different.

        Args:
            query: The search query text
            k: Number of results to return
            include_opposite: Whether to also find most different events

        Returns:
            Dictionary with 'similar' and optionally 'different' results,
            each containing list of (event, score) tuples
        """
        # Create dense query embedding and normalize
        query_dense = self.model.encode([query], convert_to_tensor=True)
        query_dense = query_dense.cpu().numpy()
        faiss.normalize_L2(query_dense)

        # Convert to binary values (0s and 1s)
        query_binary = (query_dense > 0).astype(np.uint8)

        # Pad if needed
        if query_binary.shape[1] < self.binary_dimension:
            pad_width = ((0, 0), (0, self.binary_dimension - query_binary.shape[1]))
            query_binary = np.pad(query_binary, pad_width, mode="constant")

        # Pack into bytes for FAISS
        query_packed = np.packbits(query_binary, axis=1)

        results = {}

        # Search for most similar (smallest Hamming distance)
        distances, indices = self.index.search(query_packed, k)

        # Convert Hamming distances to similarity scores
        # Maximum Hamming distance is the number of bits (binary_dimension)
        max_distance = self.binary_dimension
        similarity_scores = 1 - (distances[0] / max_distance)

        # Collect similar results
        results["similar"] = [
            (self.events[idx], float(score))
            for idx, score in zip(indices[0], similarity_scores)
            if idx in self.events
        ]

        if include_opposite:
            # For most different results, flip all bits in query
            opposite_query = ~query_packed
            distances, indices = self.index.search(opposite_query, k)
            similarity_scores = 1 - (distances[0] / max_distance)

            results["different"] = [
                (self.events[idx], float(score))
                for idx, score in zip(indices[0], similarity_scores)
                if idx in self.events
            ]

        return results


def format_event_output(
    event: NostrEvent, score: float, max_content_length: int = 200
) -> str:
    # Truncate content if too long
    content = event.content
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."

    # Convert timestamp to human-readable date
    from datetime import datetime

    date_str = datetime.fromtimestamp(event.created_at).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    # Create vector visualization if available
    vector_viz = ""
    if hasattr(event, "binary_vector") and event.binary_vector is not None:
        vector_viz = "\nBinary Vector:\n" + "".join(
            "██" if bit else "  " for bit in event.binary_vector.flatten()[:384]
        )

    return (
        f"Similarity Score: {score:.4f}\n"
        f"Event ID: {event.id}\n"
        f"Content: {content}\n"
        f"Created: {date_str}\n"
        f"Author: {event.pubkey}"
        f"{vector_viz}\n" + "-" * 80
    )


def main(
    query: str,
    nEvents: int = 1000,
    relays: List[str] = [
        "wss://relay.damus.io",
        "wss://nos.lol",
        "wss://relay.nostrplebs.com",
        "wss://relay.nostr.band",
    ],
    nPrint: int = 10,
):
    """
    Main function demonstrating binary vector search with Nostr events.

    This function:
    1. Initializes the database with the proper embedding model
    2. Fetches recent events from specified Nostr relays
    3. Creates binary vector embeddings for semantic search
    4. Performs a search for both similar and different events
    5. Displays the results in a readable format

    Args:
        query: The search text to find similar/different events
        nEvents: Number of recent events to fetch from relays
        relays: List of Nostr relay URLs to fetch from
        nPrint: Number of results to display for each category
    """
    try:
        # logger.info(f"Initializing with parameters: nEvents={nEvents}, relays={relays}")
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-xsmall-v1")
        dimension = model.get_sentence_embedding_dimension()
        db = NostrBinaryVectorDB(dimension=dimension)

        # logger.info("Fetching events...")
        events = db.fetch_events(limit=nEvents, relays=relays)
        if not events:
            logger.error("No events were fetched")
            return

        db.add_events(events)

        logger.info("Performing search...")
        results = db.search(query, k=nPrint, include_opposite=False)

        print("\n=== Most Similar Events (Based on Hamming Distance) ===")
        for event, score in results["similar"][::-1]:
            print(f"\n{format_event_output(event, score)}")

        # print("\n=== Most Different Events (Based on Hamming Distance) ===")
        # for event, score in results["different"]:
        #     print(f"\n{format_event_output(event, score)}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Full traceback: {''.join(traceback.format_tb(e.__traceback__))}")
        raise


if __name__ == "__main__":
    t0 = time()
    main(
        query="austrian economics",
        nEvents=2000,
        nPrint=20,
        relays=[
            "wss://relay.damus.io",
            "wss://relay.nostr.band",
            "wss://relay.nostrplebs.com",
            "wss://theforest.nostr1.com",
            "wss://relay.primal.net",
        ],
    )

    t1 = time()
    print(f"Time taken: {t1-t0:.2f} seconds")
