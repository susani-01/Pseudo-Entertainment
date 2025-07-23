"""
음악 Workflow의 상태를 정의하는 모듈

이 모듈은 음악 기반 콘텐츠 생성을 위한 Workflow에서 사용되는 상태 정보를 정의합니다.
LangGraph의 상태 관리를 위한 클래스를 포함합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict,Optional,Any,List,Dict

from langgraph.graph.message import add_messages


@dataclass
class MusicState(TypedDict):
    """
    음악 Workflow의 상태를 정의하는 TypedDict 클래스

    음악 기반 콘텐츠 생성을 위한 Workflow에서 사용되는 상태 정보를 정의합니다.
    LangGraph의 상태 관리를 위한 클래스로, Workflow 내에서 처리되는 데이터의 형태와 구조를 지정합니다.
    """
    #Trigger input
    theme: Optional[str]
    tags: Optional[str]  
    
    #Relevance Filter
    search_terms:List[str]
    content_hints:List[str]
    target_mood:str

    #crawling
    raw_content: Dict[str, Any]
    
    #Preprocessing
    clean_content:Dict[str, Any]
    
    #Semantic Analysis
    analysis: Dict[str,Any]
    
    #Mcp transformation
    mcp_output: Dict[str,Any]
    
    #Generation
    creative_output:Dict[str,Any]
    
    #langgraph system fields
    messages: Annotated[List[Any],add_messages]

initial_state: MusicState = {
    "theme": "",
    "tags": None,
    "search_terms":[],
    "content_hints":["poems"],
    "target_mood":"neutral",
    "raw_content":{},
    "clean_content":{"clean_text":"","is_valid":False},
    "analysis":{},
    "mcp_output":{},
    "creative_output":{},
    "messages":[]
          
}
