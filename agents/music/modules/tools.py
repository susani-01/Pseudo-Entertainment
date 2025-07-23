"""
도구(Tools) 모듈

이 모듈은 LangGraph Workflow에서 사용할 수 있는 다양한 도구를 정의합니다.
도구는 LLM이 외부 시스템과 상호작용하거나 특정 작업을 수행할 수 있도록 해주는 함수들입니다.

현재는 주석 처리된 예시 코드만 포함되어 있으며, 필요에 따라 실제 구현을 추가할 수 있습니다.
아래 예시 코드는 음악 관련 정보를 검색하는 도구를 보여줍니다.

추후 개발 시 다음과 같은 다양한 도구를 구현하여 추가할 수 있습니다:
- 음악 정보 검색 도구: 음악 장르, 아티스트, 앨범 등에 대한 정보 검색
- 음악 추천 도구: 사용자 취향에 맞는 음악 추천
- 오디오 분석 도구: 음악 파일의 특성 분석
- 음악 생성 도구: AI를 활용한 음악 생성
- 음악 변환 도구: 음악 파일 형식 변환 및 처리
"""

from typing import Any, Callable, List, Optional, Union, Dict
import logging
import httpx
import json
from pathlib import Path
import re
from collections import defaultdict
import asyncio
from collections.abc import Awaitable
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class EnhancedRelevanceFilter:
    """Trigger Processing"""
    
    def __init__(self, config_path: str = "theme_config.json"):
        self.seed_rules = self._load_config(config_path)
        self.dynamic_rules = {}
        self.rule_cache = defaultdict(dict)
        self.session_stats = defaultdict(int)
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            with open(Path(path), 'r', encoding='utf-8') as f:
                config = json.load(f)
            return {
                k.replace('_', ' '): v 
                for k, v in config.items()
                if isinstance(v, dict)
            }
        except Exception as e:
            logger.warning(f"Config load failed, using defaults: {e}")
            return {
                "healing from loss": {
                    "keywords": ["healing", "grief", "recovery"],
                    "content_hints": ["poems", "eulogies"],
                    "default_mood": "bittersweet"
                }
            }

    async def process_trigger(self, theme: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert triggers to search parameters with caching"""
        tags = tags or []
        cache_key = f"{theme}-{'-'.join(sorted(tags or []))}"
        
        if cache_key in self.rule_cache:
            self.session_stats["cache_hits"] += 1
            return self.rule_cache[cache_key]
        
        # Generate new rules
        rules = await self._generate_rules(theme.lower(), tags)
        self.rule_cache[cache_key] = rules
        self.session_stats["processed"] += 1
        return rules

    async def _generate_rules(self, theme: str, tags: List[str]) -> Dict[str, Any]:
        """Generate rules using multiple strategies"""
        # Strategy 1: Exact match
        if theme in self.seed_rules:
            return self._apply_seed_rules(theme, tags)
            
        # Strategy 2: Partial match
        for known_theme, rules in self.seed_rules.items():
            if known_theme in theme:
                return self._apply_partial_match(known_theme, theme, tags)
                
        # Strategy 3: Dynamic generation
        return {
            "keywords": self._extract_keywords(theme, tags),
            "content_hints": ["general"],
            "default_mood": "neutral",
            "is_dynamic": True
        }

    def _apply_seed_rules(self, theme: str, tags: List[str]) -> Dict[str, Any]:
        """Apply predefined rules"""
        rules = self.seed_rules[theme].copy()
        if tags:
            rules["keywords"].extend(self._process_tags(tags))
        return rules

    def _apply_partial_match(self, known_theme: str, new_theme: str, tags: List[str]) -> Dict[str, Any]:
        """Handle similar themes"""
        base = self.seed_rules[known_theme].copy()
        new_keywords = list(set(new_theme.split()) - set(known_theme.split()))
        base["keywords"].extend(new_keywords)
        if tags:
            base["keywords"].extend(self._process_tags(tags))
        base["is_derived"] = True
        return base

    def _process_tags(self, tags: List[str]) -> List[str]:
        """Normalize and expand tags"""
        return [
            processed
            for tag in tags
            for processed in self._expand_tag(tag.lower())
            if len(processed) > 3
        ]

    def _expand_tag(self, tag: str) -> List[str]:
        """Tag-specific expansions"""
        expansions = {
            "memor": ["memory", "memorial"],
            "sad": ["sadness", "grief"]
        }
        for prefix, words in expansions.items():
            if tag.startswith(prefix):
                return words
        return [tag]

    def _extract_keywords(self, text: str, tags: List[str]) -> List[str]:
        """Extract meaningful keywords"""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        return list(set(words + self._process_tags(tags or [])))

class RobustStoryCrawler:
    """Stage 3: Reliable Content Retrieval"""
    
    def __init__(self, timeout: float = 30.0):
        self.client = httpx.AsyncClient(timeout=timeout)
        self.retry_config = {
            'max_attempts': 3,
            'delay': 1.0,
            'backoff': 2.0
        }
        
    async def fetch_content(self, content_type: str, **params) -> Dict[str, Any]:
        """Unified content fetching interface"""
        try:
            handler = getattr(self, f'_fetch_{content_type}')
            return await self._retry_operation(handler, **params)
        except AttributeError:
            return {"error": f"Unsupported content type: {content_type}"}
        except Exception as e:
            logger.error(f"Content fetch failed: {str(e)}")
            return {"error": "Content unavailable"}

    async def _fetch_poems(self, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch a poem from PoetryDB, trying keywords individually for better results."""

        # 1. Handle the case where no keywords are provided by fetching a random poem.
        if not keywords:
            url = "https://poetrydb.org/random"
            response = await self.client.get(url)
            response.raise_for_status()  # Will raise an exception for HTTP errors
            data = response.json()
            if isinstance(data, list) and data:
                return {
                    "type": "poems",
                    "content": self._format_poem(data[0]),
                    "source": url,
                    "keywords": []
                }
            raise ValueError("Random poem API returned invalid data.")

        # 2. If keywords are provided, iterate through them until a match is found.
        for keyword in keywords:
            clean_keyword = keyword.strip()
            if not clean_keyword:
                continue

            url = f"https://poetrydb.org/lines/{clean_keyword}"
            try:
                response = await self.client.get(url)
                # The API can return a 200 OK status even if no poem is found.
                # In that case, the body contains a specific status message.
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "status" in data and data["status"] == 404:
                        logger.info(f"No poem found for keyword: '{clean_keyword}'")
                        continue  # Move to the next keyword.

                    # If the response is a list, a poem was found.
                    if isinstance(data, list) and data and "lines" in data[0]:
                        logger.info(f"Successfully fetched poem with keyword: '{clean_keyword}'")
                        return {
                            "type": "poems",
                            "content": self._format_poem(data[0]),  # Use the first result
                            "source": url,
                            "keywords": [clean_keyword]
                        }
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error for keyword '{clean_keyword}': {e}")
                continue  # Try the next keyword on a server error.

        # 3. If no keywords yielded a result, raise an error.
        raise ValueError("No poem found for any of the provided keywords.")

    async def _fetch_lyrics(self, artist: str, title: str) -> Dict[str, Any]:
        """Fetch lyrics from Lyrics.ovh"""
        if not artist or not title:
            raise ValueError("Missing artist/title")
            
        response = await self.client.get(
            f"https://api.lyrics.ovh/v1/{artist}/{title}"
        )
        data = response.json()
        
        return {
            "type": "lyrics",
            "content": data.get("lyrics", ""),
            "metadata": {
                "artist": artist,
                "title": title
            }
        }

    async def _retry_operation(self, operation: Callable, **kwargs) -> Any:
        """Retry logic with exponential backoff"""
        last_error = RuntimeError("No attempts were made")
        for attempt in range(self.retry_config['max_attempts']):
            try:
                return await operation(**kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.retry_config['max_attempts'] - 1:
                    delay = self.retry_config['delay'] * (self.retry_config['backoff'] ** attempt)
                    await asyncio.sleep(delay)
        raise last_error

    def _format_poem(self, poem: Dict) -> str:
        """Standardize poem format"""
        return f"{poem['title']} by {poem['author']}\n" + "\n".join(poem['lines'])

    async def close(self):
        """Cleanup resources"""
        await self.client.aclose()
class ContentPreprocessor:
    """Stage 4: Cleans and prepares raw content for analysis"""
    
    def __init__(self):
        self.noise_patterns = [
            (r"<[^>]+>", ""),          # HTML tags
            (r"\bAdvertisement\b", ""), # Ads
            (r"\s+", " ")               # Whitespace
        ]
        self.min_words = 20  # Minimum valid content length

    async def clean(self, raw: Union[str, Dict]) -> Dict[str, Any]:
        """
        Standardizes input and removes noise.
        Handles both:
        - Raw strings ("<div>Poem text...</div>")
        - API responses ({"content": "...", "meta": {...}})
        """
        # Extract text
        if isinstance(raw, dict):
            text = raw.get("content", "")
            metadata = {k: v for k, v in raw.items() if k != "content"}
        else:
            text = str(raw)
            metadata = {}

        # Apply all cleaning patterns
        for pattern, replacement in self.noise_patterns:
            text = re.sub(pattern, replacement, text)

        # Validate
        words = text.split()
        is_valid = len(words) >= self.min_words
        
        return {
            "clean_text": " ".join(words).strip(),
            "is_valid": is_valid,
            "metadata": metadata,
            "stats": {
                "word_count": len(words),
                "char_count": len(text)
            }
        }

class IntelligentSemanticAnalyzer:
    """Stage 5: Advanced Text Analysis"""
    
    def __init__(self, lexicon_path: str = "lexicon.json"):
        self.lexicon = self._load_lexicon(lexicon_path)
        self.topic_model = self._init_topic_model()
        
    def _load_lexicon(self, path: str) -> Dict[str, str]:
        """Load emotion lexicon"""
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return {
                "healing": "hopeful",
                "loss": "sad",
                "memorial": "solemn",
                "joy": "happy",
                "love": "love",
                "romance": "love",
                "affection": "love",
                "passion": "passion",
                "delight": "joy",
                "happy": "happy",
                "happiness": "happy",
                "grief": "sad",
                "sad": "sad",
                "sorrow": "sad",
                "longing": "longing",
                "fear": "fear",
                "brave": "courage",
                "hope": "hopeful",
                "serene": "calm",
                "peace": "calm",
            }

    def _init_topic_model(self) -> Dict[str, List[str]]:
        return {
            "love": ["love", "romance", "affection", "heart", "passion"],
            "friendship": ["friend", "friends", "companionship", "bond"],
            "nature": ["nature", "tree", "flower", "river", "mountain", "spring", "autumn", "summer", "winter",
                       "forest", "ocean"],
            "loss": ["loss", "grief", "mourning", "absence"],
            "recovery": ["healing", "growth", "renewal"],
            "war": ["battle", "war", "soldier", "army", "conflict"],
            "peace": ["peace", "calm", "stillness", "serenity"],
            "joy": ["joy", "happiness", "cheer", "delight", "happy"],

        }

    async def analyze(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        if not text or len(text.split()) < 5:
            return {"error": "Insufficient text for analysis"}
            
        return {
            "emotions": await self._detect_emotions(text),
            "topics": self._detect_topics(text),
            "key_phrases": self._extract_key_phrases(text),
            "analysis_metadata": {
                "word_count": len(text.split()),
                "readability": self._assess_readability(text)
            }
        }

    async def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotional tone"""
        emotions = set()
        text_lower = text.lower()
        for word, emotion in self.lexicon.items():
            if word in text_lower:
                emotions.add(emotion)
        return sorted(emotions) or ["neutral"]

    def _detect_topics(self, text: str) -> List[str]:
        """Identify dominant topics"""
        text_lower = text.lower()
        return [
            topic for topic, keywords in self.topic_model.items()
            if any(kw in text_lower for kw in keywords)
        ] or ["general"]

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract significant phrases"""
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s]
        return [
            sentences[i] 
            for i in [0, len(sentences)//2, -1] 
            if i < len(sentences)
        ][:3]

    def _assess_readability(self, text: str) -> float:
        """Estimate readability score"""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        return min(1.0, max(0.0, 1 - (avg_word_length - 4)/6))
    
class MCPClient:
    """
    Handles communication with the Metaphor-Creative-Prompt (MCP) Server
    """
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def retry_operation(self, *args, **kwargs):
        print("Retry logic running")
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self._client = None  # Will be initialized in __aenter__
        self.retry_config = {
            "stop": stop_after_attempt(3),
            "wait": wait_exponential(multiplier=1, min=2, max=10),
            "retry_error_callback": self._fallback_output
        }
        self.retry_operation = retry(**self.retry_config)(self.retry_operation)

    async def __aenter__(self):
        """Initialize the async client when entering the context"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
            headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the async client when exiting the context"""
        if self.client:
            await self.client.aclose()
    #@retry(**self.retry_config)
    async def transform_content(self, analysis: dict, workflow_id: str) -> dict:
        """
        Transforms analyzed content via MCP Server, using the provided arguments.
        """
        payload = {
            "text": analysis.get("clean_text", ""),  # Use the 'analysis' parameter
            "analysis": {
                "topics": analysis.get("key_topics", []),
                "detected_emotions": analysis.get("detected_emotions", [])
            },
            "operations": [
                "extract_metaphors",
                "build_emotional_arc",
                "generate_prompts"
            ],
            "workflow_id": workflow_id
        }

        response = await self.client.post("/v2/transform", json=payload)
        response.raise_for_status()
        raw_data = response.json()
        return self._format_output(raw_data)


    def _format_output(self, raw: dict) -> dict:
        creative_elements = raw.get("creative_elements", {})
        prompts_data = creative_elements.get("prompts", {})

        # Handle case where prompts comes as a list
        if isinstance(prompts_data, list):
            prompts_dict = {}
            for item in prompts_data:
                if isinstance(item, dict):
                    prompts_dict.update(item)
        else:
            prompts_dict = prompts_data if isinstance(prompts_data, dict) else {}

        return {
            "metadata": raw.get("metadata", {}),
            "creative_elements": {
                "metaphors": creative_elements.get("metaphors", []),
                "emotional_arc": creative_elements.get("emotions", {}),
                "prompts": {
                    "lyrics": prompts_dict.get("lyrics", ""),
                    "story": prompts_dict.get("story", "")
                }
            },
            "system_integration": raw.get("system_integration", {})
        }
    def _fallback_output(self, ctx) -> Dict[str, Any]:
        """Fallback when MCP Server fails"""
        logger.error(f"MCP failed after retries: {ctx.outcome.exception()}")
        analysis = ctx.args[0]
        
        return {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "error": "MCP_unavailable",
                "fallback": True
            },
            "creative_elements": {
                "metaphors": [],
                "emotional_arc": {
                    analysis.get("primary_emotion", "neutral"): 1.0
                },
                "prompts": {
                    "lyrics": f"Write a song about {analysis.get('key_topics', ['mom'])}",
                    "story": "Tell a story about personal transformation"
                }
            },
            "system_integration": {
                "requires_review": True
            }
        }

    async def close(self):
        """Cleanup client"""
        await self.client.aclose()

# Tool registry
AsyncToolSignature = Callable[..., Awaitable[Union[str, Dict[str, Any]]]]

# Tool registry update:
TOOLS: List[AsyncToolSignature] = [
    EnhancedRelevanceFilter().process_trigger,
    RobustStoryCrawler().fetch_content,
    ContentPreprocessor().clean,
    IntelligentSemanticAnalyzer().analyze,
    MCPClient().transform_content
]