import os
from typing import Dict, List, Any
import h5py
import numpy as np
from pathlib import Path
import base64
from PIL import Image
import io
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def resize_base64_image(base64_string: str, size=(1024, 1024)) -> str:
    """Resize a base64 encoded image."""
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def load_fashion_clip_embeddings(embeddings_file: str) -> Dict[str, Any]:
    """Load Fashion-CLIP embeddings from HDF5 file."""
    embeddings = {}
    image_paths = []
    with h5py.File(embeddings_file, 'r') as f:
        emb_group = f['embeddings']
        path_group = f['paths']
        
        # Load path mappings
        path_mapping = {i: path_group.attrs[f'path_{i}'] for i in range(len(path_group.attrs))}
        
        # Load embeddings
        for key in emb_group.keys():
            # Find original path
            original_path = next(path for path in path_mapping.values() if Path(path).stem == key)
            embeddings[original_path] = emb_group[key][:]
            image_paths.append(original_path)
    
    return embeddings, image_paths

def generate_image_summaries(image_paths: List[str]) -> List[str]:
    """Generate summaries for images using GPT-4V."""
    # Initialize GPT-4V
    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
    
    # Prompt template for image summarization
    prompt = """You are an expert fashion analyst. Describe this fashion design in detail, focusing on:
    1. Type of garment
    2. Key design elements
    3. Materials and textures
    4. Color scheme
    5. Unique features
    Be concise but comprehensive for retrieval purposes."""
    
    summaries = []
    print("Generating image summaries...")
    for img_path in tqdm(image_paths):
        # Encode and resize image
        img_base64 = encode_image(img_path)
        img_base64 = resize_base64_image(img_base64, size=(1024, 1024))
        
        # Generate summary
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        summaries.append(msg.content)
    
    return summaries

def create_multimodal_retriever(
    embeddings_file: str,
    collection_name: str = "fashion_designs"
) -> MultiVectorRetriever:
    """Create a multi-vector retriever for fashion designs."""
    # Load Fashion-CLIP embeddings
    embeddings, image_paths = load_fashion_clip_embeddings(embeddings_file)
    
    # Generate summaries for images
    image_summaries = generate_image_summaries(image_paths)
    
    # Initialize vectorstore and storage
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings()
    )
    store = InMemoryStore()
    id_key = "doc_id"
    
    # Create retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    # Add documents to vectorstore and docstore
    doc_ids = [str(i) for i in range(len(image_paths))]
    
    # Create summary documents for vectorstore
    summary_docs = [
        Document(
            page_content=summary,
            metadata={
                id_key: doc_ids[i],
                "image_path": image_paths[i],
                "embedding": embeddings[image_paths[i]].tolist()
            }
        )
        for i, summary in enumerate(image_summaries)
    ]
    
    # Add to vectorstore
    retriever.vectorstore.add_documents(summary_docs)
    
    # Create base64 encoded images for docstore
    image_docs = []
    for i, path in enumerate(image_paths):
        img_base64 = encode_image(path)
        img_base64 = resize_base64_image(img_base64, size=(1024, 1024))
        image_docs.append(img_base64)
    
    # Add to docstore
    retriever.docstore.mset(list(zip(doc_ids, image_docs)))
    
    return retriever

def split_image_text_types(docs: List[Document]) -> Dict[str, List[str]]:
    """Split retrieved documents into images and texts."""
    images = []
    texts = []
    
    for doc in docs:
        if isinstance(doc, Document):
            # Check if the content looks like a base64 string
            if isinstance(doc.page_content, str) and doc.page_content.startswith('/9j/'):
                images.append(doc.page_content)
            else:
                texts.append(doc.page_content)
        else:
            # Direct base64 string
            if isinstance(doc, str) and doc.startswith('/9j/'):
                images.append(doc)
            else:
                texts.append(doc)
    
    return {"images": images, "texts": texts}

def fashion_prompt_func(data_dict: Dict[str, Any]) -> List[HumanMessage]:
    """Create prompt for fashion analysis."""
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    
    # Add images to the messages
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    
    # Add the analysis prompt
    text_message = {
        "type": "text",
        "text": (
            "You are a fashion design expert analyzing clothing designs.\n"
            "You will be given fashion design images and their descriptions.\n"
            "Use this information to provide detailed analysis related to the user's question.\n"
            f"User question: {data_dict['question']}\n\n"
            "Design descriptions:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    
    return [HumanMessage(content=messages)]

def create_fashion_rag_chain(retriever: MultiVectorRetriever):
    """Create RAG chain for fashion analysis."""
    # Initialize GPT-4V
    model = ChatOpenAI(temperature=0, model="chatgpt-4o-latest", max_tokens=1024)
    
    # Create the RAG chain
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(fashion_prompt_func)
        | model
        | StrOutputParser()
    )
    
    return chain

def analyze_embeddings(embeddings: Dict[str, np.ndarray], image_paths: List[str]):
    """Analyze and print Fashion-CLIP embeddings."""
    print("\nFashion-CLIP Embeddings Analysis:")
    print("-" * 50)
    
    # Print embedding dimensions
    embedding_dim = next(iter(embeddings.values())).shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of images: {len(embeddings)}")
    
    # Calculate pairwise similarities
    similarities = {}
    for i, (path1, emb1) in enumerate(embeddings.items()):
        name1 = Path(path1).stem
        for path2, emb2 in list(embeddings.items())[i+1:]:
            name2 = Path(path2).stem
            similarity = float(np.dot(emb1, emb2))
            similarities[(name1, name2)] = similarity
    
    # Find most similar pairs
    top_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Most Similar Design Pairs:")
    for (name1, name2), similarity in top_pairs:
        print(f"{name1} ←→ {name2}: {similarity:.3f}")
    
    # Calculate embedding statistics
    embedding_norms = {Path(path).stem: np.linalg.norm(emb) for path, emb in embeddings.items()}
    
    print("\nEmbedding Statistics:")
    print(f"Average norm: {np.mean(list(embedding_norms.values())):.3f}")
    print(f"Std dev norm: {np.std(list(embedding_norms.values())):.3f}")
    
    # Print individual embeddings
    print("\nIndividual Embeddings:")
    for path, embedding in embeddings.items():
        name = Path(path).stem
        print(f"\n{name}:")
        print(f"Shape: {embedding.shape}")
        print(f"Norm: {np.linalg.norm(embedding):.3f}")
        print(f"Mean: {np.mean(embedding):.3f}")
        print(f"Std: {np.std(embedding):.3f}")
        print("First 5 components:", embedding[:5])
    
    # Export embeddings to CSV
    print("\nExporting embeddings to CSV...")
    embedding_data = []
    for path, embedding in embeddings.items():
        name = Path(path).stem
        row = {"design_name": name}
        row.update({f"dim_{i}": val for i, val in enumerate(embedding)})
        embedding_data.append(row)
    
    df = pd.DataFrame(embedding_data)
    os.makedirs("embeddings", exist_ok=True)
    df.to_csv("embeddings/fashion_clip_embeddings.csv", index=False)
    print("Embeddings saved to embeddings/fashion_clip_embeddings.csv")

def generate_image_description(image_path: str, model: ChatOpenAI) -> str:
    """Generate a detailed description of the fashion design using GPT-4V."""
    base64_image = encode_image(image_path)
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {
                    "type": "text",
                    "text": """Please analyze this fashion design in detail. Include:
                    1. Overall style and category
                    2. Materials and fabric properties
                    3. Color palette and patterns
                    4. Fit and silhouette
                    5. Design elements and details
                    6. Construction techniques
                    7. Potential styling options
                    8. Target market and occasions
                    Format the response as a structured analysis with clear sections."""
                }
            ]
        )
    ]
    response = model.invoke(messages)
    return response.content

def generate_description_embeddings(descriptions: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Generate embeddings for image descriptions using OpenAI embeddings."""
    embeddings_model = OpenAIEmbeddings()
    description_embeddings = {}
    
    print("\nGenerating embeddings for image descriptions...")
    for image_path, description in tqdm(descriptions.items()):
        embedding = embeddings_model.embed_query(description)
        description_embeddings[image_path] = np.array(embedding)
    
    return description_embeddings

def save_description_embeddings(embeddings: Dict[str, np.ndarray], descriptions: Dict[str, str], output_file: str):
    """Save description embeddings and their corresponding texts to H5 file."""
    with h5py.File(output_file, 'w') as f:
        # Create groups
        emb_group = f.create_group('embeddings')
        desc_group = f.create_group('descriptions')
        
        # Store embeddings and descriptions
        for path, embedding in embeddings.items():
            name = Path(path).stem
            emb_group.create_dataset(name, data=embedding)
            desc_group.create_dataset(name, data=descriptions[path].encode('utf-8'))
        
        # Store paths
        paths = list(embeddings.keys())
        f.create_dataset('paths', data=[p.encode('utf-8') for p in paths])

def create_knowledge_base(descriptions: Dict[str, str], description_embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Create a structured knowledge base from descriptions and embeddings."""
    knowledge_base = {
        'designs': {},
        'relationships': [],
        'metadata': {
            'total_designs': len(descriptions),
            'embedding_dim': next(iter(description_embeddings.values())).shape[0],
            'creation_date': datetime.now().isoformat()
        }
    }
    
    # Process each design
    for path, description in descriptions.items():
        name = Path(path).stem
        embedding = description_embeddings[path]
        
        # Calculate similarity with other designs
        similarities = []
        for other_path, other_embedding in description_embeddings.items():
            if other_path != path:
                similarity = float(np.dot(embedding, other_embedding))
                similarities.append((Path(other_path).stem, similarity))
        
        # Sort and get top similar designs
        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        
        # Store design information
        knowledge_base['designs'][name] = {
            'description': description,
            'embedding': embedding.tolist(),
            'similar_designs': top_similar,
            'path': str(path)
        }
        
        # Store relationships
        for similar_name, similarity in top_similar:
            knowledge_base['relationships'].append({
                'design1': name,
                'design2': similar_name,
                'similarity': similarity,
                'type': 'visual_similarity'
            })
    
    return knowledge_base

def save_knowledge_base(knowledge_base: Dict[str, Any], output_file: str):
    """Save the knowledge base to a JSON file."""
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_kb = knowledge_base.copy()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_kb, f, indent=2, ensure_ascii=False)
    print(f"Knowledge base saved to {output_file}")

def main():
    # Path to saved Fashion-CLIP embeddings
    embeddings_file = "embeddings/fashion_clip_embeddings.h5"
    descriptions_file = "embeddings/description_embeddings.h5"
    knowledge_base_file = "knowledge_base/fashion_knowledge_base.json"
    
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file not found at {embeddings_file}")
        print("Please run visualize_graph.py first to generate embeddings.")
        return
    
    print("Loading Fashion-CLIP embeddings...")
    embeddings, image_paths = load_fashion_clip_embeddings(embeddings_file)
    
    # Generate detailed descriptions
    print("\nGenerating detailed image descriptions...")
    model = ChatOpenAI(model="chatgpt-4o-latest", max_tokens=1000)
    descriptions = {}
    for path in tqdm(image_paths):
        descriptions[path] = generate_image_description(path, model)
    
    # Generate description embeddings
    description_embeddings = generate_description_embeddings(descriptions)
    save_description_embeddings(description_embeddings, descriptions, descriptions_file)
    print(f"\nDescription embeddings saved to {descriptions_file}")
    
    # Create and save knowledge base
    print("\nCreating knowledge base...")
    knowledge_base = create_knowledge_base(descriptions, description_embeddings)
    save_knowledge_base(knowledge_base, knowledge_base_file)
    
    # Analyze embeddings
    analyze_embeddings(embeddings, image_paths)
    print("\nDescription Embeddings Analysis:")
    analyze_embeddings(description_embeddings, image_paths)

if __name__ == "__main__":
    main() 