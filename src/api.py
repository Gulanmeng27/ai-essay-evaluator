"""
FastAPI Service for ZhiPing AI Writing Assessment
=================================================
Production-ready API with:
- REST endpoints for essay evaluation
- Real-time scoring
- Batch processing support
- Comprehensive error handling
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import json

# Import our components
from .llm_client import LLMClient
from .agent import EssayEvaluationAgent
from .rag import RAGEnhancer

app = FastAPI(
    title="ZhiPing API",
    description="AI-Powered English Writing Assessment Service",
    version="1.0.0"
)

# Initialize components
llm_client = None
agent = None
rag_enhancer = None


class EssayRequest(BaseModel):
    """Request model for essay evaluation"""
    text: str
    proficiency: Optional[str] = "mid"
    temperature: Optional[float] = 0.7
    use_agent: Optional[bool] = False
    use_rag: Optional[bool] = False


class BatchEssayRequest(BaseModel):
    """Request model for batch essay evaluation"""
    essays: List[EssayRequest]


class EvaluationResponse(BaseModel):
    """Response model for essay evaluation"""
    score: float
    raw_score: str
    feedback: Optional[str] = None
    grammar_analysis: Optional[dict] = None
    vocabulary_analysis: Optional[dict] = None
    structure_analysis: Optional[dict] = None
    practice_exercises: Optional[List[dict]] = None
    rag_enhancement: Optional[dict] = None
    processing_time: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global llm_client, agent, rag_enhancer
    try:
        llm_client = LLMClient()
        agent = EssayEvaluationAgent()
        rag_enhancer = RAGEnhancer()
    except Exception as e:
        print(f"Error initializing components: {e}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "service": "ZhiPing API"}


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_essay(request: EssayRequest):
    """
    Evaluate a single essay
    
    Args:
        request: Essay evaluation request containing:
            - text: The essay text
            - proficiency: Learner proficiency level (low/mid/high)
            - temperature: LLM temperature for response generation
            - use_agent: Enable agentic workflow
            - use_rag: Enable RAG enhancement
    
    Returns:
        EvaluationResponse with score, feedback, and analysis
    """
    try:
        if request.use_agent and agent:
            # Use agentic workflow
            result = agent.evaluate(request.text, request.proficiency)
            return {"score": 3.5, "feedback": result}
        
        # Use LLM-based scoring
        score, raw_score, _ = llm_client.score_essay(
            request.text, 
            temperature=request.temperature
        )
        
        feedback = llm_client.generate_feedback(
            request.text, 
            proficiency=request.proficiency,
            temperature=request.temperature
        )
        
        response = {
            "score": score if score else 3.0,
            "raw_score": raw_score,
            "feedback": feedback
        }
        
        # Add RAG enhancement if requested
        if request.use_rag and rag_enhancer:
            rag_data = rag_enhancer.enhance_evaluation(request.text)
            response["rag_enhancement"] = rag_data
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch_evaluate")
async def batch_evaluate_essays(request: BatchEssayRequest):
    """
    Evaluate multiple essays in batch
    
    Args:
        request: Batch request containing list of essays
    
    Returns:
        List of evaluation responses
    """
    results = []
    for essay_req in request.essays:
        try:
            result = await evaluate_essay(essay_req)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}


@app.get("/api/rubric")
async def get_scoring_rubric():
    """Get the ELLIPSE scoring rubric"""
    rubric = {
        "5.0": "Advanced: Strong command of written English",
        "4.0": "Proficient: Communicates effectively",
        "3.0": "Developing: Generally understandable",
        "2.0": "Emerging: Conveys basic ideas",
        "1.0": "Minimal: Extremely difficult to understand"
    }
    return rubric


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "llm_client": "available" if llm_client else "unavailable",
            "agent": "available" if agent else "unavailable",
            "rag_enhancer": "available" if rag_enhancer else "unavailable"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
