
import os

from dotenv import load_dotenv
from hdbcli import dbapi

# HANASparqlQAChain ties together : Schema aware SPARQL Generation
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_hana import HanaRdfGraph, HanaSparqlQAChain



# Load environment variables if needed
load_dotenv()

# Establish connection to SAP HANA Cloud
connection = dbapi.connect(
    address=os.environ.get("HANA_DB_ADDRESS"),
    port=os.environ.get("HANA_DB_PORT"),
    user=os.environ.get("HANA_DB_USER"),
    password=os.environ.get("HANA_DB_PASSWORD"),
    autocommit=True,
    sslValidateCertificate=False,
)



graph_uri = "workforce"

graph = HanaRdfGraph(
    connection=connection,
    graph_uri=graph_uri,
    auto_extract_ontology=True
)


#Initialize LLN
llm = ChatOpenAI(proxy_model_name="gpt-4o", temperature=0)

# Initialize the Graph with Custom SPARQL Prompt and Custom QA Prompt
qa_chain = HanaSparqlQAChain.from_llm(
    llm=llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True,
    sparql_generation_prompt=YOUR_SPARQL_PROMPT,
    qa_prompt=YOUR_QA_PROMPT
)

