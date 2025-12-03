pip install langchain_hana

import os

from dotenv import load_dotenv
from hdbcli import dbapi
from langchain_hana import HanaRdfGraph

# Load environment variables if needed
load_dotenv()

# Establish connection to SAP HANA Cloud
connection = dbapi.connect(
    address=os.environ.get("HANA_DB_ADDRESS"),
    port=os.environ.get("HANA_DB_PORT"),
    user=os.environ.get("HANA_DB_USER"),
    password=os.environ.get("HANA_DB_PASSWORD"),
    autocommit=True,
    # sslValidateCertificate=False,
)



# Pass the .ttl file created during the execution of createOnotology.py

graph = HanaRdfGraph(
    connection=connection,
    ontology_local_file="<your_ontology_file_path>", # e.g., "ontology.ttl"
    ontology_local_file_format="<your_ontology_file_format>",  # e.g., "Turtle", "RDF/XML", "JSON-LD", "N-Triples", "Notation-3", "Trig", "Trix", "N-Quads"
)

#This function will return the graph object after pushing the .ttl file into the HANA DB

