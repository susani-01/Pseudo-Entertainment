"""LangChain 체인을 설정하는 함수 모듈

LCEL(LangChain Expression Language)을 사용하여 체인을 구성합니다.
기본적으로 modules.prompt 템플릿과 modules.models 모듈을 사용하여 LangChain 체인을 생성합니다.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from agents.music.modules.models import get_openai_model

# Initialize LLM
llm = get_openai_model(model_name="gpt-4", temperature=0.6)

# Define prompt template
music_prompt_template = PromptTemplate.from_template("""
You're given two creative text inputs:

Poem Metadata:
{poem}

Lyrics Metadata:
{lyrics}

Task: Create a reflective, unified narrative or analysis that combines the mood, emotion, and story of both inputs.
Genre: {music_genre}
Mood: {music_mood}
""")

def get_music_generation_chain():
    """Return a LangChain chain that operates on the unified content state."""
    def format_inputs(state: dict) -> dict:
        return {
            "poem": state.get("poem", {}).get("mcp", ""),
            "lyrics": state.get("lyrics", {}).get("mcp", ""),
            "music_genre": state.get("music_genre", ""),
            "music_mood": state.get("music_mood", "")
        }

    return RunnableLambda(format_inputs) | music_prompt_template | llm
