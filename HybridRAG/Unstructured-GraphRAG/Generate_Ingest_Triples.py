# Scenario3: Knowledge Graph Construction from PDF Documents using AWS Bedrock and SAP HANA Cloud
#Required packages: Make Sure you install packages mentioned below. And the code is tested end to end using colab
#%pip install rdflib langchain_core langchain_experimental langchain_community langchain_text_splitters 
#%pip install  langchain_openai pypdf hdbcli langchain-aws boto3

# Importing the Python regex module for regular expression operations
import re

# Importing Document class from langchain_core for handling document objects
from langchain_core.documents import Document

# Importing Graph Transformer for converting text to graph structures
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Importing Text Splitter for processing and chunking text documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importing ThreadPoolExecutor for parallel processing of document chunks
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Importing PDF loader for processing PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Importing TokenTextSplitter for splitting text by tokens rather than characters
from langchain_text_splitters import TokenTextSplitter

# Importing RDFlib components for working with RDF graphs
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF  # RDF namespace for basic RDF properties

# Importing operating system module for file path operations
import os

# Importing SAP HANA Database API for connecting to HANA Cloud
from hdbcli import dbapi

# Importing AWS Bedrock dependencies for LLM integration
from langchain_aws import ChatBedrock
import boto3

# Importing base classes for creating custom language model configuration
from langchain_core.language_models.base import BaseLanguageModel
from pydantic import BaseModel, Field
import boto3

# Custom configuration class for AWS Bedrock LLM
class CustomBedrockLLMConfig(BaseModel):
    model_id: str = Field(..., description="ID of the Bedrock model to be used")
    aws_access_key_id: str = Field(..., description="AWS IAM access key for authentication")
    aws_secret_access_key: str = Field(..., description="AWS IAM secret key for authentication")
    aws_region_name: str = Field(..., description="AWS region where Bedrock is available")

# Custom implementation of BaseLanguageModel for AWS Bedrock
class CustomBedrockLLM(BaseLanguageModel):
    config: CustomBedrockLLMConfig  # Configuration holder

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = CustomBedrockLLMConfig(**kwargs)  # Initialize config

    def _call(self, query):
        # Create a session with the provided AWS credentials
        session = boto3.Session(
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
            region_name=self.config.aws_region_name
        )

        # Create a Bedrock client using the established session
        bedrock_client = session.client('bedrock')

        # Invoke the Bedrock model with the provided query
        response = bedrock_client.invoke_model(
            ModelId=self.config.model_id,
            Input=query
        )

        return response['Output']  # Return the model's output

    # Below are various standard interface methods required by LangChain
    def __call__(self, query):
        return self._call(query)

    def agenerate_prompt(self, input_prompt):
        return input_prompt

    def apredict(self, query):
        return self._call(query)

    def apredict_messages(self, messages):
        return self._call(messages[0].content)

    def generate_prompt(self, input_prompt):
        return input_prompt

    def invoke(self, query):
        return self._call(query)

    def predict(self, query):
        return self._call(query)

    def predict_messages(self, messages):
        return self._call(messages[0].content)

# Define a namespace for our RDF graph with a unique URI
# This will be used as the base URI for all nodes in the knowledge graph
EX = Namespace("http://finalboto_check_test_mission_faqhanahotspots.org/")

# Establish connection to SAP HANA Cloud database
conn = dbapi.connect(
    user = "<You HANA Cloud User Name>", 
    password = "<Your HANA Cloud Password>",
    address = '<Your HANA Cloud host>',
    port = 443,
)

# AWS Bedrock Configuration - credentials for accessing AWS services

AWS_ACCESS_KEY_ID = "XXXXXXXXXXXX" # AWS IAM access key (truncated for security)
AWS_SECRET_ACCESS_KEY = "XXXXXXXXXXXX"# AWS IAM secret key (truncated for security)
AWS_DEFAULT_REGION = "us-east-1"# AWS region where Bedrock is available

# Initialize Bedrock runtime client for invoking models
bedrock_client = boto3.client(
    service_name='bedrock-runtime',  # Bedrock runtime service
    aws_access_key_id=AWS_ACCESS_KEY_ID,  # AWS credentials
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Initialize Claude LLM through LangChain's ChatBedrock interface
claude_llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Specific model version
    client=bedrock_client,  # Configured Bedrock client
    model_kwargs={
        "temperature": 0.5,  # Controls randomness (0=deterministic, 1=creative)
        "max_tokens": 2048  # Maximum tokens to generate
    }
)

# Instantiate an empty RDF Graph to store our knowledge graph
g = Graph()

# Create LLM Graph Transformer instance that will convert text to graph structures
llm_transformer = LLMGraphTransformer(
    llm=claude_llm,  # Using Claude LLM for graph generation
)

# Function to load PDF documents from specified path
def load_documents():
    documents = []  # List to hold loaded documents
    file_path = "//content//pdf//2927209_E_20250327.pdf"  # Path to PDF file

    try:
        # Create PDF loader instance for the specified file
        loader = PyPDFLoader(file_path)
        # Load and parse the PDF document
        loaded_docs = loader.load()
        if loaded_docs:
            # Add loaded documents to our collection
            documents.extend(loaded_docs)
            print(loaded_docs)  # Debug print of loaded documents
    except Exception as e:
        print(f"Error loading documents: {e}")  # Error handling

    return documents  # Return list of loaded documents

# Function to clean text by removing unwanted characters
def clean_text(text):
    bad_chars = ['"', "\n", "'"]  # Characters to remove
    for char in bad_chars:
        text = text.replace(char, '')  # Remove each bad character
    return text  # Return cleaned text

# Function to split documents into manageable chunks
def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    # Create token-based text splitter with specified chunk size and overlap
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,  # Maximum tokens per chunk
        chunk_overlap=chunk_overlap  # Tokens overlapping between chunks
    )

    chunks = []  # List to hold document chunks
    for doc in documents:
        # Clean the document text before chunking
        cleaned_text = clean_text(doc.page_content)
        # Split the cleaned text into chunks
        doc_chunks = text_splitter.split_text(cleaned_text)
        # Create Document objects for each chunk with original metadata
        for chunk in doc_chunks:
            chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunks  # Return list of document chunks

# Main function for processing documents into knowledge graph
def process_documents(llm_transformer):
    # Load documents from PDF file
    documents = load_documents()

    if not documents:
        print("No documents loaded.")  # Early return if no documents
        return []

    # Split documents into chunks for processing
    chunks = create_chunks(documents)
    print(f"Documents split into {len(chunks)} chunks.")

    graph_document_list = []  # List to hold processed graph documents

    # Use ThreadPoolExecutor for parallel processing of chunks
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []  # List to hold future objects

        # Submit each chunk for parallel processing
        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(llm_transformer.convert_to_graph_documents, [chunk]))

        # Process completed futures as they finish
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                # Get the result of the processing
                graph_document = future.result()
                # Add to our collection of graph documents
                graph_document_list.extend(graph_document)
                print(f"Chunk {i} processed into graph document.")
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

    return graph_document_list  # Return list of processed graph documents

# Helper function to create safe URIs from strings
def safe_uri(string):
    # Replace spaces and slashes with underscores for URI safety
    return URIRef(EX[string.replace(" ", "_").replace("/", "_")])

# Process the documents and convert them into graph documents
graph_documents = process_documents(llm_transformer)

# Process each graph document to extract nodes and relationships
for document in graph_documents:
    # Process each node in the document
    for node in document.nodes:
        # Create safe URIs for node and its type
        node_uri = safe_uri(node.id)
        node_type_uri = safe_uri(node.type)

        # Add node type triple to the RDF graph
        g.add((node_uri, RDF.type, node_type_uri))

        # Add all properties of the node as triples
        for key, value in node.properties.items():
            g.add((node_uri, safe_uri(key), Literal(value)))

    # Process each relationship in the document
    for relationship in document.relationships:
        # Create URIs for source, target and relationship type
        source_uri = safe_uri(relationship.source.id)
        target_uri = safe_uri(relationship.target.id)
        relationship_type_uri = safe_uri(relationship.type)

        # Add relationship triple to the RDF graph
        g.add((source_uri, relationship_type_uri, target_uri))

# Convert RDF graph to a list of triples (subject, predicate, object)
rdf_triples = []
for s, p, o in g:
    rdf_triples.append((str(s), str(p), str(o)))

print(rdf_triples)  # Print all generated triples for debugging

triples = []  # List to hold formatted triples for SPARQL

# Create database cursor for executing SPARQL queries
cursor = conn.cursor()
try:
    # Format each triple for SPARQL INSERT statement
    for s, p, o in rdf_triples:
        triples.append(f"<{str(s)}> <{str(p)}> <{str(o)}>")

    # Combine all triples into a single SPARQL INSERT DATA statement
    triples_str = " .\n    ".join(triples) + " ."
    sparql_insert_query = f"INSERT DATA {{ \n    {triples_str} \n}}"

    # Execute the SPARQL query using HANA's stored procedure
    resp = cursor.callproc('SPARQL_EXECUTE', (sparql_insert_query, 
              "Accept: application/sparql-results+xml Content-Type: application/sparql-query", 
              '?', None))

    # Extract response metadata and query response
    metadata_headers = resp[3]  # Response headers from SPARQL execution
    query_response = resp[2]    # Actual query response

    # Print success message with namespace information
    print("\nTriples successfully generated in SAP HANA Cloud.")
    print(f"Query the KGs with this namespace from SAP HANA Cloud DB Explorer: {EX}")

    # Debugging information
    print("Inserted Triple:", (s, p, o))
    print("Response Metadata:", metadata_headers)
    print("Query Response:", query_response)

finally:
    cursor.close()  # Ensure cursor is closed even if errors occur