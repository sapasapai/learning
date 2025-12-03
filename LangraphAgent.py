#Create Langgraph Agent so it will give 2 options 1. Ingest Data (press 1) , Query Data (Press 2) , and ask the user to input query and give result
# it will be a simple agents, while True:
    choice = input("Enter 1 to Ingest Data or 2 to Query Data: ")
    if choice == '1':
        # Code to ingest data
        print("Ingesting data...")
        # Add your ingest data logic here
    elif choice == '2':
        query = input("Please enter your query: ")
        # Code to query data
        print(f"Querying data for: {query}")
        # Add your query data logic here
    else:
        print("Invalid choice. Please try again.")

        