import ollama
import pandas as pd
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
import subprocess
import uuid
from enum import Enum
import datetime
from dataclasses import dataclass, field
from prompt_classes import ResponseType, LLMResponse

STC = Namespace("http://w3c.org/stc/core#")
CI = Namespace("http://w3c.org/stc/copyright-infringement#")


# Load the TTL file into a graph
def load_turtle_file(file_path: str) -> Graph:
    """
    Load a TTL file into an RDFLib Graph.
    
    Args:
        file_path (str): The path to the Turtle file.
        
    Returns:
        Graph: The loaded RDF graph.
    """
    graph = Graph()
    try:
        graph.parse(file_path, format="turtle")
        print(f"Graph loaded successfully. Contains {len(graph)} triples.")
    except Exception as e:
        print(f"Error loading graph: {e}")
    return graph

# Base class for the Ollama server
class OllamaServer:
    def __init__(self, client):
        """
        Initialize the OllamaServer instance with the provided client and a Graph to store the Ollama server info.
        Args:
            client: An instance of the Ollama client that provides access to the server.
        """
        self.client = client
        self.graph = Graph()
        self.graph.bind("ci", CI)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("stc", STC)

    def get_uri(self) -> URIRef:
        """Generate a unique URI for this instance."""
        return STC[f"{self.__class__.__name__}_{str(uuid.uuid4())}"]

    def get_models_details(self):
        """
        Return the list of the LLMs that are currently stored on the cluster.
        Returns:
            list: A list of model navailable on the Ollama server with all their details.
        """
        try:
            models = self.client.list()
            return models.get("models", [])
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def get_models_list(self):
        """
        Return the simplified list of the LLMs that are currently stored on the cluster.
        Returns:
            list: A list of model names available on the Ollama server.
        """
        try:
            models = self.get_models_details()
            names = [m.model for m in models]
            return names
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def download_model_if_not_exists(self, targeted_model: str):
        models = self.get_models_details()
        names = [m.model for m in models]
        names = [m.replace(":latest","") for m in names]
        if targeted_model not in names:
            subprocess.run(f"ollama pull {targeted_model}", shell=True, check=True, stderr=subprocess.STDOUT)
        else:
            print(f"{targeted_model} is already available in the server")
            self.client.show(targeted_model)

    def models_to_rdf(self, graph) -> URIRef:
        """
        Convert the list of models to RDF triples.
        Args:
            models (list): A list of model names.
        Returns:
            Graph: An RDF graph containing triples describing the models on the server.
        """
        models = self.get_models_details()
        try:
            for m in models:
                # Create a URI for each model
                uri = self.get_uri()
                graph.add((uri, RDF.type, STC.LLM)) # STC core ontology LLM class
                graph.add((uri,STC.hasAPIName,Literal(m.model.replace(":latest",""), datatype=XSD.string)))
                graph.add((uri,STC.downloadedFromAPIAtDate,Literal(m.modified_at, datatype=XSD.dateTimeStamp)))
                graph.add((uri,STC.hasLLMFamily,Literal(m.details.family, datatype=XSD.string)))
                graph.add((uri,STC.hasParameterSize,Literal(m.details.parameter_size, datatype=XSD.string)))
                graph.add((uri,STC.hasQuantizationLevel,Literal(m.details.quantization_level, datatype=XSD.string)))
        except Exception as e:
            print(f"Error creating RDF triples: {e}")
            return None

    def save(self, file_path: str):
        """Save the info about the models that are on the server to a file in Turtle format."""
        self.models_to_rdf(self.graph)
        self.graph.serialize(destination=file_path, format="turtle")


class OllamaChat:
    USER = 'user'  # Role for user messages
    ASSISTANT = 'assistant'  # Role for assistant (LLM) messages

    def __init__(self, server: OllamaServer, model: str):
        """
        Initialize the OllamaChat instance with the provided server and model.
        
        Args:
            server (OllamaServer): The server instance to interact with.
            model (str): The model name to use for prompts.
        """
        self.server = server
        self.model = model
        self.messages = []  # List of dictionaries with 'role' and 'content'
        #self.answers_rdf =

        # Ensure the model is available
        self.server.download_model_if_not_exists(model)

    def add_history(self, content: str, role: str):
        """
        Add a message to the history.
        
        Args:
            content (str): The message content.
            role (str): The role of the message sender ('user' or 'assistant').
        """
        self.messages.append({'role': role, 'content': content})

    def get_history(self):
        """
        Retrieve the entire history of messages.

        Returns:
            list: A list of dictionaries containing the conversation history. 
                  Each dictionary has 'role' (user/assistant) and 'content' keys.
        """
        return self.messages
    
    def clear_history(self):
        """Clear the conversation history."""
        self.messages = []

    def send_prompt(self, prompt: str, prompt_uuid: str, use_history: bool = True, stream: bool = False) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt (str): The prompt to send.
            use_history (bool): Do the LLM has to use the previous prompts and response to answer the current prompt ?
            stream (bool): Whether to stream the response.

        Returns:
            str: The response from the LLM.
        """
        try:
            if use_history:
                # Combine the history into a single input
                context = "\n".join([f": {q}\nA: {a}" for q, a in self.messages])
                full_prompt = f"PREVIOUS MESSAGES{context}\nQ: {prompt}"
            else:
                full_prompt = prompt
                
            # Add the user's prompt to the history
            self.add_history(prompt, self.USER)

            # Send the prompt to the server
            response = self.server.client.chat(model=self.model, messages=self.messages, stream=stream)

            # Parse the response
            complete_message = ''
            if stream:
                for line in response:
                    message_content = line['message']['content']
                    complete_message += message_content
                    print(message_content, end='', flush=True)
            else:
                complete_message = response.get('message', {}).get('content', '').strip()

            # Add the assistant's response to the history
            self.add_history(complete_message, self.ASSISTANT)

            # Create and return an LLMResponse instance
            response_instance = LLMResponse(
                prompt_id=str(uuid.uuid4()),  # Have to be replaced by the input prompt uuid
                raw_text=complete_message,
                timestamp=datetime.datetime.now(),
                response_type=ResponseType.GENERATED
            )
            return response_instance
        except Exception as e:
            print(f"Error during prompt execution: {e}")
            return LLMResponse(
                prompt_id=str(uuid.uuid4()), # Have to be replaced by the input prompt uuid
                raw_text=f"Error: {str(e)}",
                timestamp=datetime.datetime.now(),
                response_type=ResponseType.ERROR
            )