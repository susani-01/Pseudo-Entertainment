"""
노드 클래스 모듈

해당 클래스 모듈은 각각 노드 클래스가 BaseNode를 상속받아 노드 클래스를 구현하는 모듈입니다.
"""

from agents.base_node import BaseNode
from agents.music.modules.tools import (
                                        MCPClient,
                                        IntelligentSemanticAnalyzer
                                        )
from agents.music.modules.models import get_openai_model
from agents.music.modules.chains import get_music_generation_chain
from langchain_core.messages import SystemMessage,HumanMessage
from typing import List, Optional, Union,Dict,Any
import logging
import yaml

logger = logging.getLogger(__name__)

async def content_fetch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input schema for the langsmith studio .

    Example Input
    {
      "theme": "love",
      "tags": ["romance", "affection", "passion"],
      "target_mood": "romantic"
    }
    """

    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, dict) and last_message.get("content"):
            try:
                new_input = yaml.safe_load(last_message["content"])
                if isinstance(new_input, dict):
                    state.update(new_input)
            except (yaml.YAMLError, TypeError):
                state["messages"].append("Error: Invalid input format. Please use valid JSON or YAML.")
                return state

    #  Input validation
    if not state.get("theme"):
        state["messages"].append("Error: No theme provided")
        state["raw_content"] = {"text": "", "metadata": {"error": "missing_theme"}}
        return state

    # Prepare search parameters (optional, to make prompts nicer)
    try:
        raw_keywords = generate_keywords(state["theme"], state.get("tags", []))
        state["search_terms"] = [k.strip().lower() for k in raw_keywords if k.strip()]
        state["content_hints"] = ["lyrics" if state.get("target_mood") in ["energetic", "happy"] else "poems"]
    except Exception as e:
        state["messages"].append(f"Parameter error: {str(e)}")
        state["raw_content"] = {"text": "", "metadata": {"error": "parameter_error"}}
        return state

    # Generate a poem with OpenAI
    try:
        theme = state["theme"]
        mood = state.get("target_mood", "contemplative")
        tags = ", ".join(state.get("tags", []))
        author_hint = f" in the style of {tags}" if tags else ""
        poem_prompt = f"Write a {mood} poem about '{theme}'{author_hint}. The poem should be expressive and imaginative."

        llm = get_openai_model()
        response = await llm.ainvoke(poem_prompt)
        poem_text = response.content if hasattr(response, "content") else str(response)

        state["raw_content"] = {
            "text": poem_text,
            "metadata": {
                "source": "openai",
                "theme": theme,
                "mood": mood,
                "tags": state.get("tags", [])
            }
        }
    except Exception as e:
        state["messages"].append(f"OpenAI poem generation failed: {str(e)}")
        state["raw_content"] = {"text": "", "metadata": {"error": "openai_poem_failed"}}
    return state

def generate_keywords(theme: str, tags: Optional[Union[str, List[str]]]) -> List[str]:
    """Converts user input into searchable keywords."""

    # If no tags are provided, create default search terms.
    if not tags:
        return [theme, f"{theme} emotions", f"songs about {theme}"]

    # If tags are already a list (which they are from the UI), use them directly.
    if isinstance(tags, list):
        return [theme] + tags

    # As a fallback, if tags are a string, split them.
    return [theme] + tags.split(',')

# ---  Preprocessing Node ---
async def preprocess_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 4: Clean and validate content"""
    from agents.music.modules.tools import ContentPreprocessor

    raw_content = state.get("raw_content", {})

    if not raw_content or not raw_content.get("text"):
        state["messages"].append("No content to preprocess")
        return state

    preprocessor = ContentPreprocessor()
    cleaned = await preprocessor.clean(raw_content["text"])


    if not cleaned.get("is_valid"):
        state["messages"].append("Content is too short or invalid.")
        return state

    state["clean_content"] = {
        "clean_text": cleaned["clean_text"],
        "is_valid": cleaned["is_valid"]
    }
    return state

#   Analysis Node
async def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 5: Semantic analysis"""
    # Check if clean_content exists and has clean_text
    if "clean_content" not in state or not state["clean_content"].get("clean_text"):
        state["messages"].append("Error: Missing cleaned content.")
        # Ensure analysis key exists to avoid crashing downstream
        state["analysis"] = {
            "clean_text": "",
            "key_topics": [],
            "detected_emotions": [],
            "intent": "unknown",
            "key_phrases": [],
            "readability": 0.0
        }
        return state

    analyzer = IntelligentSemanticAnalyzer()
    result = await analyzer.analyze(state["clean_content"]["clean_text"])

    if "error" in result:
        state["messages"].append(result["error"])
        # Still provide minimal structure
        state["analysis"] = {
            "clean_text": state["clean_content"]["clean_text"],
            "key_topics": [],
            "detected_emotions": [],
            "intent": "unknown",
            "key_phrases": [],
            "readability": 0.0
        }
        return state

    # If successful, store full result
    state["analysis"] = {
        "clean_text": state["clean_content"]["clean_text"],
        "key_topics": result.get("topics", []),
        "detected_emotions": result.get("emotions", ["neutral"]),
        "intent": result.get("intent", "inform"),
        "key_phrases": result.get("key_phrases", []),
        "readability": result.get("analysis_metadata", {}).get("readability", 0.5)
    }
    return state


# ---  MCP Transformation Node ---
async def mcp_transform_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Extracts data from state and calls the MCP tool correctly."""

    #  Extract the analysis data from the state
    analysis_data = state.get("analysis", {})
    if not analysis_data or not analysis_data.get("clean_text"):
        state.setdefault("messages",[]).append("Error Missing analysis data for MCP transform")
        return state

    #  Extract the workflow ID from the state.
    workflow_id = state.get("thread_id", "unknown-workflow")

    #  Call the tool with BOTH required arguments, using their names.
    try:
        async with MCPClient() as client:
            state["mcp_output"] = await client.transform_content(
                analysis=analysis_data,
                workflow_id=workflow_id
            )

    except Exception as e:
        state.setdefault("messages", []).append(f"MCP call failed: {str(e)}")
        state["error"] = str(e)
    return state

async def langchain_music_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final pipeline step: use LangChain chain to merge poem + lyrics MCP outputs
    into a unified narrative/creative text.
    """
    try:
        chain = get_music_generation_chain()
        result = await chain.ainvoke(state)
        state["final_narrative"] = result  # store chain's output (string)
    except Exception as e:
        logger.error(f"LangChain synthesis failed: {e}")
        state["error"] = str(e)
    return state

class MusicGenerationNode(BaseNode):
    """
    음악 장르와 분위기에 적합한 음악을 생성하는 노드
    """

    async def execute(self, state: dict) -> dict:
        try:
            prompt_text = state["mcp_output"]["creative_elements"]["prompts"]["lyrics"]
            if not prompt_text:
                raise ValueError("Received an empty lyrics prompt from MCP.")

            # Get an instance of the OpenAI model
            llm = get_openai_model()

            # Prepare the messages for the LangChain model
            messages = [
                SystemMessage(
                    content=f"You are a creative songwriter. The user wants a song with a {state.get('music_mood', 'general')} mood in the {state.get('music_genre', 'pop')} genre."
                ),
                HumanMessage(content=prompt_text)
            ]

            # Invoke the model asynchronously and get the response
            response = await llm.ainvoke(messages)

            # The response is an AIMessage object;
            lyrics = response.content

            state["creative_output"] = {
                "lyrics": lyrics,
                "metadata": state["mcp_output"]["creative_elements"]
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state["error"] = str(e)
        return state

    



