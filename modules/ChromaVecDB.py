import pandas as pd
import chromadb
from tqdm import tqdm
from chromadb.utils import embedding_functions


class ChromaVecDB():
    name = "documents"
    
    def __init__(self, model, path = None, metadata={"hnsw:space": "cosine"}, name = "documents"):
        if path==None:
            self.client = chromadb.Client()
        else:
            self.client = chromadb.PersistentClient(path)
        self.metadata=metadata
        self.model=model
        self.name = name
        self.get_collection()
    
    def get_collection(self, name = None):
        if name == None:
            name = self.name
        return self.client.get_or_create_collection(
            name,
            metadata=self.metadata
            )
    
    def empty_collection(self, name = None):
        if name == None:
            name = self.name
        try:
            self.client.delete_collection(name)
            self.get_collection()
        except ValueError:
            pass
    
    def store_concepts(self, df:pd.DataFrame, batch_size = 20000):
        """
        store concepts along with its embeddings

        Args:
            df (pd.DataFrame): a concept dataframe with `concept_name` and `concept_id` columns
            embeddings: A matrix, the embeddings for each concept, the row number must match the row number of `df`
            batch_size (int, optional): Batch size. Defaults to 20000.
        """
        collection = self.get_collection()
        df = df.reset_index()
        concept_names = df.concept_name.to_list()
        concept_ids = [str(i) for i in df.concept_id.tolist()]
        for i in tqdm(range(0, len(df), batch_size)):
            batch_documents = concept_names[i:i + batch_size]
            batch_ids = concept_ids[i:i + batch_size]
            #batch_embeddings = embeddings[i:i + batch_size]
            batch_embeddings = self.embed(df[i:i + batch_size], show_progress_bar=False)
            collection.add(
                documents=batch_documents,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
    
    def embed(self, df, batch_size=1024, show_progress_bar=True):
        """
        Embed concept from a dataframe

        Args:
            df (pd.DataFrame): a concept dataframe with `concept_name`
        """
        return self.model.encode(df.concept_name.tolist(), show_progress_bar = show_progress_bar, batch_size=batch_size)
    
    def query(self, df, n_results=100, batch_size=1024, show_progress_bar=True):
        """
        Query the database with a dataframe of concepts and return the closest matches
        
        Args:
            df (pd.DataFrame): a concept dataframe with `concept_name`
            n_results (int, optional): Number of results to return. Defaults to 100.
        """
        embeddings = self.embed(df, batch_size=batch_size, show_progress_bar=show_progress_bar)
        return self._query(embeddings, n_results=n_results, batch_size=batch_size)
    
    def _query(self, embeddings, n_results=100, batch_size=1024):
        collection = self.get_collection()
        all_results = None
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Querying in batches"):
            embeddings_dt = embeddings[i:i + batch_size]
            res = collection.query(
                query_embeddings=embeddings_dt,
                n_results=n_results,
                include = ['distances']
            )
            if i==0:
                all_results = res
            else:
                ## append res dict element to all_results
                for key in res:
                    if all_results[key] is not None:
                        all_results[key].extend(res[key])
        all_results['ids'] = [[int(i) for i in x] for x in all_results['ids']]
        return all_results
    
    def set_doc_name(self, doc_name):
        self.name = doc_name
    
    def get_doc_name(self):
        return self.name
        
        
