import os
import faiss
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from src.utils.utils import load_dataframes


class SeasonSpecificRAG:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.indices = {}  # Dictionary to store season-specific FAISS indices
        self.rag_data = {}  # Dictionary to store season-specific datasets

    def prepare_rag_data(self, data_dir):
        # Load and process data from CSV files
        dfs = load_dataframes(data_dir)

        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            rag_entries = []
            for _, row in season_df.iterrows():
                if row['type'] == 'player':
                    entry = f"Player: {row['name']}\nPosition: {row['position']}\nClub: {row['club']}\n"
                    entry += f"Performance:\n  Goals: {row['goals']}\n  Assists: {row['assists']}"
                elif row['type'] == 'club':
                    entry = f"Club: {row['name']}\nChampions League Titles: {row['cl_titles']}\n"
                    entry += f"CL Performance: {row['cl_performance']}"
                # Add similar processing for games and events
                rag_entries.append(entry)

            self.rag_data[season] = Dataset.from_dict({"text": rag_entries})

    def build_indices(self):
        for season, dataset in self.rag_data.items():
            embeddings = self.embedding_model.encode(dataset['text'])
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype('float32'))
            self.indices[season] = index

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # Save the RAG datasets
        for season, dataset in self.rag_data.items():
            dataset.save_to_disk(os.path.join(output_dir, f'rag_dataset_{season}'))

        # Save the FAISS indices
        for season, index in self.indices.items():
            faiss.write_index(index, os.path.join(output_dir, f'rag_index_{season}.faiss'))

        # Save the embedding model
        self.embedding_model.save(os.path.join(output_dir, 'embedding_model'))

        # Save the list of seasons
        with open(os.path.join(output_dir, 'seasons.txt'), 'w') as f:
            for season in self.rag_data.keys():
                f.write(f"{season}\n")

    @classmethod
    def load(cls, input_dir: str):
        instance = cls()

        # Load the embedding model
        instance.embedding_model = SentenceTransformer(os.path.join(input_dir, 'embedding_model'))

        # Load the list of seasons
        with open(os.path.join(input_dir, 'seasons.txt'), 'r') as f:
            seasons = [line.strip() for line in f]

        # Load the RAG datasets and FAISS indices
        for season in seasons:
            instance.rag_data[season] = Dataset.load_from_disk(os.path.join(input_dir, f'rag_dataset_{season}'))
            instance.indices[season] = faiss.read_index(os.path.join(input_dir, f'rag_index_{season}.faiss'))

        return instance

    def retrieve_relevant_info(self, query: str, season: str, k: int = 5) -> List[str]:
        if season not in self.indices:
            raise ValueError(f"No data available for season {season}")

        query_embedding = self.embedding_model.encode([query])
        _, indices = self.indices[season].search(query_embedding.astype('float32'), k)
        return [self.rag_data[season]['text'][i] for i in indices[0]]


# Usage example
rag_system = SeasonSpecificRAG()
rag_system.prepare_rag_data('path/to/your/data')
rag_system.build_indices()
rag_system.save('path/to/save/rag/data')

# Later, in your main application
loaded_rag = SeasonSpecificRAG.load('path/to/save/rag/data')

query = "Top scorers in the Premier League"
season = "2022/2023"
relevant_info = loaded_rag.retrieve_relevant_info(query, season)
print(f"Relevant information for '{query}' in season {season}:")
for info in relevant_info:
    print(info)
print("\n")


# Using in FantasyFootballRAGChunked
class FantasyFootballRAGChunked:
    def __init__(self, model, tokenizer, rag_system: SeasonSpecificRAG):
        self.model = model
        self.tokenizer = tokenizer
        self.rag_system = rag_system

    def generate_team_chunked(self, prompt: str, season: str) -> str:
        relevant_info = self.rag_system.retrieve_relevant_info(prompt, season)
        augmented_prompt = f"{prompt}\nSeason: {season}\n\nRelevant Information:\n" + "\n".join(relevant_info)

        inputs = self.tokenizer(augmented_prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(**inputs, max_length=2048)
        return self.tokenizer.decode(outputs[0])


# Usage in main script
rag_system = SeasonSpecificRAG.load('path/to/save/rag/data')
fantasy_rag = FantasyFootballRAGChunked(fine_tuned_model, tokenizer, rag_system)

prompt = "Create a fantasy team with the best performers"
season = "2022/2023"
response = fantasy_rag.generate_team_chunked(prompt, season)
print(response)