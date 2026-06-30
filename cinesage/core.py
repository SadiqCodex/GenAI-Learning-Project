from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Use local Ollama phi3 model via the ollama CLI
import subprocess


class Movie(BaseModel):
    title: str 
    release_year : Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str



parser = PydanticOutputParser(pydantic_object=Movie)


prompt = ChatPromptTemplate.from_messages([
    ('system',"""
Extract movie information from the paragraph
     {format_instructions}
"""),
("human","{paragraph}")
]
)



para = input("Give your paragraph : ")

final_prompt = prompt.invoke(
    {"paragraph" : para,
     'format_instructions': parser.get_format_instructions()
     }
)

# ensure prompt is a plain string
prompt_text = final_prompt if isinstance(final_prompt, str) else getattr(final_prompt, 'content', str(final_prompt))

# call ollama chat (requires ollama installed and model 'phi3' available locally)
proc = subprocess.run(["ollama", "chat", "phi3", prompt_text], capture_output=True, text=True)
if proc.returncode != 0:
    raise RuntimeError(f"Ollama chat failed: {proc.stderr}")

response_text = proc.stdout.strip()
movie_data = parser.parse(response_text)

print(movie_data)
