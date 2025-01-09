
import requests

# Define the UMLS API base URL and your API key

class UMLS_API:
    API_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    def __init__(self, api_key):
        self.api_key = api_key
        
    def umls_get(self, url, query={}):
        query.update({"apiKey": self.api_key})
        response = requests.get(url, params=query)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")
        
        response.encoding = 'utf-8'
        return response.json()

    def get_data_by_source(self, vocabulary, source_code):
        url = f"{self.API_BASE_URL}/content/current/source/{vocabulary}/{source_code}"
        data = self.umls_get(url)
        return data

    def get_atom_by_source(self, vocabulary, source_code):
        url = f"{self.API_BASE_URL}/content/current/source/{vocabulary}/{source_code}/atoms"
        data = self.umls_get(url)
        return data

    def search_by_source(self, vocabulary, source_code):
        url = f"{self.API_BASE_URL}/search/current"
        query = {"string": source_code, "sabs": vocabulary, "inputType": "sourceConcept", "returnIdType": "concept"}
        data = self.umls_get(url, query)
        return data

    def get_definitions_by_CUI(self, CUI):
        url = f"{self.API_BASE_URL}/content/current/CUI/{CUI}/definitions"
        data = self.umls_get(url)
        values = [i['value'] for i in data['result']]
        return values

    def get_definitions_by_source(self, vocabulary, source_code):
        data = self.search_by_source(vocabulary, source_code)
        CUIs = [i['ui'] for i in data['result']['results']]
        definitions = [self.get_definitions_by_CUI(CUI) for CUI in CUIs]
        definitions = [item for sublist in definitions for item in sublist]
        return definitions
