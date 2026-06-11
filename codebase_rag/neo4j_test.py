import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

# Load variables from .env file
load_dotenv()

class Neo4jApp:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        
        # Enforce defaults if the environment variables return None
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "12345678")
        
        if not password:
            raise ValueError("Critical Error: NEO4J_PASSWORD variable is empty or missing.")
            
        # Pack the credentials into a structured tuple
        self.auth = (username, password)
        self.driver = None

    def connect(self):
        """Initializes the database driver and verifies the network connection."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j database.")
        except DriverError as e:
            print(f"Driver failed to connect to {self.uri}: {e}")
            raise

    def close(self):
        """Closes the driver instance and releases all pooled connections."""
        if self.driver:
            self.driver.close()
            print("Neo4j driver connection closed.")

    def create_person(self, name: str, age: int):
        """Executes a write transaction to create a node in the graph."""
        # Using execute_query provides automatic transactional fallback and bookmarking
        query = """
        MERGE (p:Person {name: $name})
        SET p.age = $age
        RETURN p.name AS name, p.age AS age
        """
        try:
            # execute_query automatically opens and closes sessions
            records, summary, keys = self.driver.execute_query(
                query,
                name=name,
                age=age,
                database_="neo4j"
            )
            for record in records:
                print(f"Created/Updated Person: {record['name']} (Age: {record['age']})")
        except Neo4jError as e:
            print(f"Database error executing query: {e}")

    def create_friendship(self, name1: str, name2: str):
        """Creates a relationship between two existing Person nodes."""
        query = """
        MATCH (a:Person {name: $name1})
        MATCH (b:Person {name: $name2})
        MERGE (a)-[r:FRIEND_OF]->(b)
        RETURN a.name, type(r), b.name
        """
        try:
            self.driver.execute_query(query, name1=name1, name2=name2, database_="neo4j")
            print(f"Relationship created: {name1} -> FRIEND_OF -> {name2}")
        except Neo4jError as e:
            print(f"Failed to create relationship: {e}")

    def get_all_people(self):
        """Executes a read query to fetch node properties from the graph."""
        query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age"
        records, _, _ = self.driver.execute_query(query, database_="neo4j")
        return [{"name": r["name"], "age": r["age"]} for r in records]


# Application execution block
if __name__ == "__main__":
    app = Neo4jApp()
    try:
        app.connect()
        
        # Ingest nodes and relationships
        app.create_person("Alice", 30)
        app.create_person("Bob", 25)
        app.create_friendship("Alice", "Bob")
        
        # Read the state of the graph
        people = app.get_all_people()
        print(f"Current People in Database: {people}")
        
    finally:
        app.close()
