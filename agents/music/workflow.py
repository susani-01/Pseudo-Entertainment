"""
음악 관련 콘텐츠 생성을 위한 Workflow 모듈

이 모듈은 음악 기반 콘텐츠 생성을 위한 Workflow를 정의합니다.
StateGraph를 사용하여 음악 처리를 위한 워크플로우를 구축합니다.
"""

from langgraph.graph import StateGraph
from agents.base_workflow import BaseWorkflow
from agents.music.modules.state import MusicState
from agents.music.modules.nodes import MusicGenerationNode
from langchain_core.tracers import LangChainTracer
from agents.music.modules.nodes import (
    content_fetch_node,
    preprocess_node,
    analyze_node,
    mcp_transform_node,
    langchain_music_node,
    MusicGenerationNode
)
from agents.music.modules.tools import (
    EnhancedRelevanceFilter,
    RobustStoryCrawler,
    ContentPreprocessor,
    IntelligentSemanticAnalyzer,
    MCPClient
)

def validate_state(state):
    if "music_genre" not in state or "music_mood" not in state:
        raise ValueError("Missing required music parameters")
    return state


class MusicWorkflow(BaseWorkflow):
    """
    음악 관련 콘텐츠 생성을 위한 Workflow 클래스

    이 클래스는 음악 기반 콘텐츠 생성을 위한 Workflow를 정의합니다.
    BaseWorkflow를 상속받아 기본 구조를 구현하고, MusicState를 사용하여 상태를 관리합니다.
    """

    def __init__(self, state: MusicState):
        super().__init__()
        self.state = state
        self.name = "music_workflow"
        self.tools = {
            
            "trigger_filter": EnhancedRelevanceFilter().process_trigger,
            "fetch_content": RobustStoryCrawler().fetch_content,
            "clean_content": ContentPreprocessor().clean,
            "analyze_content": IntelligentSemanticAnalyzer().analyze,
            "transform_content": MCPClient().transform_content
            
        }

    def build(self):
        """
        음악 Workflow 그래프 구축 메서드

        StateGraph를 사용하여 음악 처리를 위한 Workflow 그래프를 구축합니다.
        현재는 음악 생성 노드를 포함하고 있으며, 추후 조건부 에지를 추가하여
        다양한 경로를 가진 Workflow를 구축할 수 있습니다.

        Returns:
            CompiledStateGraph: 컴파일된 상태 그래프 객체
        """

        #조건부 에지 추가 예시
        # builder.add_conditional_edges(
        #     "call_model",
        #     # call_model 실행이 완료된 후, 다음 노드(들)는
        #     # router의 출력을 기반으로 예약됩니다
        #     router,
        # )
        builder = StateGraph(self.state)
        
        #add nodes
        builder.add_node("fetch_content", content_fetch_node)
        builder.add_node("preprocess", preprocess_node)
        builder.add_node("analyze", analyze_node)
        builder.add_node("mcp_transform", mcp_transform_node)
        builder.add_node("generate_music", MusicGenerationNode().execute)
        builder.add_node("synthesize_narrative", langchain_music_node)
        
        # Workflow
        builder.add_edge("__start__", "fetch_content")
        builder.add_edge("fetch_content", "preprocess")
        builder.add_edge("preprocess", "analyze")
        builder.add_edge("analyze", "mcp_transform")
        builder.add_edge("mcp_transform", "generate_music")
        builder.add_edge("generate_music", "synthesize_narrative")
        builder.add_edge("synthesize_narrative", "__end__")

                         
        workflow = builder.compile()  # 그래프 컴파일
        workflow.name = self.name  # Workflow 이름 설정
        return workflow
    
  

# 음악 Workflow 인스턴스 생성
music_workflow = MusicWorkflow(MusicState)
