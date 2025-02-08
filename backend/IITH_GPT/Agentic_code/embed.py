import faiss
import json
import os
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

def initialize_faiss_index(dim):
    index = faiss.IndexFlatL2(dim) 
    return index

def processjson(json_file, index, metadata):
    with open(json_file, 'r', encoding='utf-8', errors='ignore') as file:
        print(file)
        data = json.load(file)
    
    for entry in data:
        title = entry.get('title', '')
        content = ' '.join(entry.get('content', []))
        
        name_without_extension = os.path.splitext(os.path.basename(json_file))[0]

        title_embedding = model.encode([title])
        content_embedding = model.encode([content])
        file_embedding = model.encode(name_without_extension)

        title_embedding = torch.tensor(title_embedding).to(device)
        content_embedding = torch.tensor(content_embedding).to(device)
        file_embedding = torch.tensor(file_embedding).to(device)
   
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        sections = text_splitter.split_text(content)
        
        section_embeddings = []
        for section in sections:
            section_embedding = model.encode([section])
            section_embedding = torch.tensor(section_embedding).to(device)
            section_embeddings.append(section_embedding.cpu().numpy())  

        title_embedding_flat = title_embedding.flatten().cpu().numpy()
        section_embeddings_flat = [embedding.flatten() for embedding in section_embeddings]

        combined_embedding = (
            1.2 * file_embedding.cpu().numpy() + 
            1.2 * title_embedding_flat + 
            1 * sum(section_embeddings_flat)  
        ) / (2.4 + len(section_embeddings_flat))

        index.add(combined_embedding.reshape(1, -1)) 
        
        metadata.append({
            "main":name_without_extension,
            "title": title,
            "sections": sections
        })

directory_path = '../data'
output_directory = '../output_multilevel_index'

os.makedirs(output_directory, exist_ok=True)

index = initialize_faiss_index(384)  
metadata = []

# Process all JSON files in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(directory_path, file_name)
        
        processjson(file_path, index, metadata)
        print(f"Processed {file_name} and embeddings added to FAISS index.")

index_file_path = os.path.join(output_directory, 'faiss_index.index')
faiss.write_index(index, index_file_path)
print(f"FAISS index saved to {index_file_path}")

metadata_file_path = os.path.join(output_directory, 'metadata.json')
with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
    json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
print(f"Metadata saved to {metadata_file_path}")