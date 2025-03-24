
import pandas as pd
from string import Template
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import uuid
import datetime

class RDFMappable(ABC):
    """Abstract base class for objects that can be mapped to RDF."""
    @abstractmethod
    def to_rdf(self, graph: Graph) -> URIRef:
        pass

    def get_uri(self) -> URIRef:
        """Generate a unique URI for this instance."""
        return CI[f"{self.__class__.__name__}_{str(uuid.uuid4())}"]

# Response Type Enum
class ResponseType(Enum):
    REFUSAL = "refusal"
    GENERATED = "generated"
    MIXED = "mixed"

    
@dataclass(kw_only=True)
class LLMResponse(RDFMappable):
    """Base class for all LLM responses."""
    prompt_id: str
    raw_text: str = field(default="")
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    response_type: ResponseType = field(default=ResponseType.GENERATED)
    
    def to_rdf(self, graph: Graph) -> URIRef:
        uri = self.get_uri()
        graph.add((uri, RDF.type, CI.LLMResponse))
        graph.add((uri, CI.promptId, Literal(self.prompt_id)))
        graph.add((uri, CI.timestamp, Literal(self.timestamp)))
        graph.add((uri, CI.responseType, Literal(self.response_type.value)))
        graph.add((uri, CI.rawText, Literal(self.raw_text)))
        return uri

class PromptCreation:


    def __init__(self, path_prompt="/Users/marco/Documents/GitHub/EMNLP-MultiPRIDE/script/dataset_input/test_prompts.csv"):

        self.config=pd.read_csv(path_prompt, sep=";")

    def prompt_creation(self, prompt_method="Baseline", samples=[["Attacca","Meglio fascista che frocio? Sceglie fra due disgrazie. Sono contro l'omofobia, ma i froci non hanno mai mandato in Russia 67k mila ragazzi senza equipaggiamento, dei quali solo 7k tornati ma congelati, fra cui mio nonno. E' fiera di suo nonno che ha fatto questo  e .... vergogna!"],["Sostiene","@Rosi33998582 @mps274 No. Questo è un insulto lesbica no LGBT sta per Lesbica Gay Bisessuali Trans usare la parola froc*o ha una connotazione dispregiativa come lo è diventata neg*o che invece prima si usava ( se sbaglio correggetemi per favore )."]], language="Ita", task=None):

        #this function create the prompt according the parameters provide:

        #prompt_method=["Baseline", "CoT", "CARP"]
        #samples= list of sample in the form ["label","text"]
        #language=["Eng","Ita","Spa"]
        #task=["text_postion_classification"]

        row = self.config[(self.config['Langauge'] == language) & (self.config['Method'] == prompt_method)]

        prompt=row['Context']+" "+row['Question']+" "+row['Output_format']+ "\n\n"

        if(len(samples)>0):

            for i in samples:

                prompt+="INPUT: "+i[1]+"\n"
                prompt+="LABEL: "+i[0]+"\n\n"

        
        prompt+="INPUT: {}"+"\n"
        prompt+="LABEL: "+"\n"

        self.prompt_output=prompt
        
        return prompt











