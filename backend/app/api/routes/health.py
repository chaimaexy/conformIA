from fastapi import APIRouter

from app.core.config import get_app_config, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    cfg = get_app_config()
    return {
        "status": "ok",
        "environment": settings.environment,
        "llm_model": cfg.llm.chat_model,
        "embedding_model": cfg.embeddings.model,
        "mistral_configured": bool(settings.mistral_api_key),
        "supabase_configured": bool(settings.supabase_url and settings.supabase_anon_key),
    }
