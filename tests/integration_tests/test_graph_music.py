import pytest
from langsmith import unit
from agents.music.workflow import MusicWorkflow
from agents.music.modules.state import MusicState

@pytest.mark.asyncio
@unit

async def test_music_workflow_end_to_end():
    initial_state = {
        "query": "Create jazz music with chill mood",
        "music_genre": "jazz",
        "music_mood": "chill",
        "response": [],
        "poem": {},
        "lyrics": {}
    }
    
    workflow=  MusicWorkflow(MusicState).build()
    result = await workflow.ainvoke(initial_state)
    
    assert result["poem"].get("text")
    assert result["lyrics"].get("text")
    assert result["response"]