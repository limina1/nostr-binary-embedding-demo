# Nostr Vector Database Example

This repository provides a minimal working example of creating and querying a vector database using Nostr events.

The example shows how to:
- Fetch events from Nostr relays using NAK
- Convert text content into semantic embeddings
- Store and search vectors efficiently using FAISS
- Compare similar and contrasting content in the vector space

### Prerequisites

You'll need Python 3.7+ and the following dependencies:

```bash
pip install sentence-transformers faiss-cpu torch numpy
```

You'll also need to install [nak](https://github.com/fiatjaf/nak/)


### Basic Usage

Here's a simple example of how to use the vector database:

```python
from nostr_vector_db import NostrVectorDB

# Initialize the database
db = NostrVectorDB()

# Fetch events from specific relays
events = db.fetch_events(limit=100, relays=["wss://relay.damus.io"])

# Add events to the vector database
db.add_events(events)

# Search for similar and different events
results = db.search("Bitcoin price prediction", k=5)

# Print results
for event, score in results['similar']:
    print(f"Similar - Score: {score:.4f}")
    print(f"Content: {event.content[:200]}...")
```

## How It Works

1. **Event Fetching**: The code uses NAK to fetch kind 1 (text) events from specified Nostr relays.

2. **Embedding Creation**: Text content is converted into semantic vectors using the `sentence-transformers` library with the "all-MiniLM-L6-v2" model. This model transforms text into 384-dimensional vectors that capture semantic meaning.

3. **Vector Storage**: The FAISS library efficiently stores and indexes these vectors, enabling fast similarity searches.

4. **Similarity Search**: When performing a search:
   - Similarity search can be seen as an optimized and approximate [K Nearest Neighbors](https://scikit-learn.org/1.5/modules/neighbors.html) search for vectors
   - The query text is converted into the same vector space
   - FAISS finds the most similar vectors using a distance metric - typically cosine similarity, for binary vectors, Hamming distance is used
   - The code can also find the most semantically different content by searching for opposite vectors

# Possible enhancements
- Compare performance/retrieval of multiple models with different dimensions
- Implement mixbread.ai's demo for nostr events
# Interesting links
- https://emschwartz.me/binary-vector-embeddings-are-so-cool/
- https://www.mixedbread.ai/blog/binary-mrl (implementation to search through 41 million Wikipedia articles using our state-of-the-art binary embeddings. More advanced, can use for another demo)
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


