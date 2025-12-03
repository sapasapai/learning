# Scenario6: In this scenario, we are planning to build KGs for Tabular Data.
# We are using SFLIGHT Schema tables(SBOOK, SCUSTOM) and build an Ontology. This will basically provide 
# semantic relation between these tables, join conditions, columns used etc. This information will help 
# us build KGs by importing the Ontology to SAP HANA CLoud. We could either import using local file or using Cloud Storages
#Required packages: Make Sure you install packages mentioned below. And the code is tested end to end using colab
#Here are the necessary Packages
#!pip install rdflib
# Import necessary libraries and modules
# Import required RDFLib components
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD

# Main function to generate Turtle (TTL) file
def generate_ttl():
    # Define custom namespaces for our RDF graph
    ns = Namespace("http://scb.uk/workforce/")  # Namespace for workforce data
    db = Namespace("http://scb.uk/database/")  # Namespace for database schema
    
    # Create an empty RDF graph
    g = Graph()
    
    # Bind namespace prefixes for cleaner serialization
    g.bind("worforce", ns)  # Associates "worforce" prefix with our namespace
    g.bind("db", db)       # Associates "db" prefix with database namespace

    # Define table resources
    semployee = ns.Employee      # Resource for bookings table
    scostcenters = ns.CostCenters  # Resource for customers table
    spayroll = ns.PAYROLL  # Resource for payroll table


    # Add metadata for Employee table
    g.add((semployee, RDF.type, db.Table))            # Set type as Table
    g.add((semployee, RDFS.label, Literal("Employee Records, Employee Table")))  # Human-readable label
    g.add((semployee, db.tableName, Literal("Employee"))) # Actual table name in database

    # Add metadata for CostCenter table
    g.add((scostcenters, RDF.type, db.Table))            # Set type as Table
    g.add((scostcenters, RDFS.label, Literal("Cost Center Records, Cost Center Table")))  # Human-readable label
    g.add((scostcenters, db.tableName, Literal("CostCenters"))) # Actual table name in database


    # Add metadata for Payroll table
    g.add((spayroll, RDF.type, db.Table))            # Set type as Table
    g.add((spayroll, RDFS.label, Literal("Payroll Table, a table containing payroll information for employees")))  # Human-readable label
    g.add((spayroll, db.tableName, Literal("Payroll"))) # Actual table name in database


    # Define columns and metadata for SBOOK table
    semployee_columns = {
        # Client column metadata
        ns.EmpID: {
            "label": "Employee ID",
            "isKey": True,  # Mark as primary key
            "description": "Employee ID, a unique id assigned to each employee"
        },
        # Carrier ID column metadata
        ns.FirstName: {
            "label": "FirstName",
            "description": "Airline carrier identifier"
        },
        # Connection ID column metadata
        ns.LastName: {
            "label": "Connection ID", 
            "description": "Flight connection identifier"
        },
        # Booking ID column metadata
        ns.Gender: {
            "label": "Booking ID",
            "isKey": True,  # Part of composite key
            "aggregation": db.COUNT,  # Can be used with COUNT function
            "description": "Unique booking identifier"
        },
        # Customer ID column metadata (foreign key)
        ns.Country: {
            "label": "Customer ID",
            "isKey": True,  # Part of composite key
            "foreignKey": ns.ID,  # References SCUSTOM.ID
            "description": "Foreign key to SCUSTOM.ID"
        },
        # Price column metadata
        ns.HireDate: {
            "label": "Price",
            "aggregation": db.SUM,  # Can be used with SUM function
            "description": "Booking price in local currency"
        },
        # Class column metadata
        ns.CostCenterID: {
            "label": "Class",
            "description": "Travel class (Economy/Business)"
        },
        # Order date column metadata
        ns.BaseSalary: {
            "label": "Booking Date",
            "filter": "TO_INT(LEFT({column}, 4))",  # Example filter transformation
            "aggregation": db.COUNT,  # Can be used with COUNT function
            "description": "Booking date (VARCHAR format)",
            "dataType": XSD.string  # Explicit data type
        },

        ns.Currency: {
            "label": "Booking Date",
            "filter": "TO_INT(LEFT({column}, 4))",  # Example filter transformation
            "aggregation": db.COUNT,  # Can be used with COUNT function
            "description": "Booking date (VARCHAR format)",
            "dataType": XSD.string  # Explicit data type
        }
    }

    # Add all SBOOK columns to the graph
    for col, meta in sbook_columns.items():
        # Basic column metadata
        g.add((col, RDF.type, db.Column))  # Set type as Column
        g.add((col, RDFS.label, Literal(meta["label"])))  # Human-readable label
        g.add((col, db.columnName, Literal(col.split("/")[-1])))  # Extract column name from URI
        g.add((col, db.description, Literal(meta["description"])))  # Description
        
        # Conditional metadata additions
        if meta.get("isKey"):
            g.add((col, db.isPrimaryKey, Literal(True)))  # Mark as primary key
        if meta.get("groupBy"):
            g.add((col, db.groupBy, Literal(True)))  # Mark as groupable
        if meta.get("aggregation"):
            g.add((col, db.aggregationFunction, meta["aggregation"]))  # Add aggregation function
        if meta.get("filter"):
            g.add((col, db.filterFunction, Literal(meta["filter"])))  # Add filter function
        if meta.get("foreignKey"):
            g.add((col, db.foreignKey, meta["foreignKey"]))  # Add foreign key reference
        if meta.get("dataType"):
            g.add((col, db.dataType, meta["dataType"]))  # Add explicit data type

    # Define columns and metadata for SCUSTOM table
    scustom_columns = {
        # Customer ID column metadata
        ns.ID: {
            "label": "Customer ID",
            "isKey": True,  # Primary key
            "description": "Primary key for customer"
        },
        # Customer name column metadata
        ns.NAME: {
            "label": "Customer Name",
            "description": "Full name of customer"
        }
    }

    # Add all SCUSTOM columns to the graph
    for col, meta in scustom_columns.items():
        # Basic column metadata
        g.add((col, RDF.type, db.Column))  # Set type as Column
        g.add((col, RDFS.label, Literal(meta["label"])))  # Human-readable label
        g.add((col, db.columnName, Literal(col.split("/")[-1])))  # Extract column name from URI
        g.add((col, db.description, Literal(meta["description"])))  # Description
        
        # Conditional metadata additions
        if meta.get("isKey"):
            g.add((col, db.isPrimaryKey, Literal(True)))  # Mark as primary key



#------------------------ Define Relationship across the tables ---------------------------#

    # Define relationships between tables
    g.add((sbook, db.relatedTo, scustom))  # General relationship between tables
    
    # Explicit foreign key relationship
    g.add((ns.CUSTOMID, db.foreignKey, ns.ID))  # SBOOK.CUSTOMID → SCUSTOM.ID
    
    # Join condition for the relationship
    g.add((ns.CUSTOMID, db.joinCondition, 
           Literal("SBOOK.MANDT = SCUSTOM.MANDT AND SBOOK.CUSTOMID = SCUSTOM.ID")))

    # Serialize the graph to Turtle format
    graph_string = g.serialize(format="turtle")

    # Write the Turtle string to a file
    with open('/content/sflight_tabular.ttl', 'w') as file:
        file.write(graph_string) 
        

# Execute the function to generate the TTL file
generate_ttl()


