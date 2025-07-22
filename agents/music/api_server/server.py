from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime,timezone
from typing import List, Dict, Optional,Union,Any,Literal
from textblob import TextBlob
import spacy
from spacy.tokens import Span
import logging
from fastapi.middleware.cors import CORSMiddleware
from modules.models import get_openai_model

# Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load NLP model with custom extensions
nlp = spacy.load("en_core_web_md")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class AnalysisInput(BaseModel):
    """
        Input schema for content transformation in the server


    {
        "text": "Write a song about the fire inside a frozen heart.",
        "analysis": {
            "theme": "love",
            "tags": ["romance", "affection", "passion"],
            "target_mood": "romantic",
            "topics": ["relationships", "heartbreak"],
            "detected_emotions": ["romantic", "longing"]
        },
        "workflow_id": "optional-id"

    }
    """
    text: str
    analysis: Dict[str, Any]  # From SemanticAnalyzer
    workflow_id: Optional[str] = None


class SystemIntegration(BaseModel):

    format_version: Optional[Literal["1.0"]] = None
    compatible_services: Optional[List[Literal["lyrics_generator", "mood_playlist"]]] = None

    # Error case field
    requires_manual_review: Optional[bool] = None


class MCPOutput(BaseModel):
    metadata: Dict[str, Union[str, bool]]  # Allow both string and bool
    creative_elements: Dict[
        str,
        Union[
            List[str],
            Dict[str, Union[float, str]]  # Allow both numbers and strings
        ]
    ]
    system_integration:SystemIntegration


Span.set_extension("polarity", getter=lambda span: TextBlob(span.text).sentiment.polarity)
@app.get("/")
def root():
    return {"message": "MCP Server is running. Use /v2/transform for processing."}

@app.post("/v2/transform")
async def transform_content(data: AnalysisInput) -> MCPOutput:
    """Enhanced processing endpoint"""
    try:
        doc = nlp(data.text)
        #1. Metaphor Extraction using LLM
        llm = get_openai_model()  # Or your LLM factory/helper
        prompt = (
                "Extract all metaphors in the following poem. "
                "List only the metaphorical phrases or sentences, one per line.\n\nPoem:\n" + data.text
        )
        response = await llm.ainvoke(prompt)
        metaphors = [line.strip() for line in response.content.splitlines() if line.strip()][:5]


        # 2. Emotional Arc
        emotional_arc = {}
        for i, sent in enumerate(doc.sents):
            emotional_arc[f"segment_{i}"] = round(sent._.polarity, 2)


        key_topics = data.analysis.get("topics",[])

        if not key_topics:
            key_topics = ["life"] #use default

        detected_emotions = data.analysis.get("detected_emotions")
        primary_emotion = detected_emotions[0] if detected_emotions else "introspective"

        if metaphors:
            metaphor_phrase = f"using metaphors like '{metaphors[0]}'"
        else:
            metaphor_phrase = "with vivid imagery"

        lyrics_prompt = (
            f"Compose a {primary_emotion} song about "
            f"{', '.join(key_topics[:2])} "
            f"{metaphor_phrase}"
        )

        story_prompt = (
            f"Write a short story with a {primary_emotion} tone "
            f"about {key_topics[0]}"
        )

        return MCPOutput(
            metadata={

                "timestamp": datetime.now(timezone.utc).isoformat(),
                "workflow_id": data.workflow_id,
                "source": "mcp_v2"
            },
            creative_elements={
                "themes": key_topics,
                "emotions": emotional_arc,
                "metaphors": metaphors,
                "prompts": {
                    "lyrics":lyrics_prompt,
                    "story":story_prompt

                }
            },


            system_integration=SystemIntegration(
                format_version="1.0",
                compatible_services=["lyrics_generator","mood_playlist"]
            )
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return MCPOutput(
            metadata={
                "error": str(e),
                "fallback":True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            creative_elements={
                "themes": data.analysis.get("topics", []),
                "emotions": {"neutral": 1.0},
                "metaphors": [],

                "prompts": {
                    "lyrics": "Write a song about life experiences",
                    "story": "Tell a story about personal growth"
                }
            },
            system_integration=SystemIntegration(
                format_version="1.0",
                compatible_services=[]
            )

        )


# Keep legacy endpoint
@app.post("/process")
def legacy_process(text: str):
    """Backward-compatible endpoint"""
    return {
        "message": "Use /v2/transform for enhanced features",
        "endpoint": "/process",
        "status": "deprecated"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)