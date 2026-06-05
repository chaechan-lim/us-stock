"""Self-evolution recommendations API.

Surfaces AgentRecommendation rows so the operator can accept/reject
proposed config changes from agents (trade_review for now).
"""

from datetime import datetime

from core.timeutil import now_utc_naive
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import desc, select

from core.models import AgentRecommendation

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecommendationDTO(BaseModel):
    id: int
    created_at: str
    agent_type: str
    param_path: str
    current_value: Any
    proposed_value: Any
    rationale: str | None
    expected_effect: str | None
    confidence: str | None
    risk: str | None
    backtest_result: Any
    status: str
    applied_at: str | None
    rejected_reason: str | None
    notes: str | None


class RejectBody(BaseModel):
    reason: str | None = None


class AcceptBody(BaseModel):
    notes: str | None = None


def _to_dto(r: AgentRecommendation) -> RecommendationDTO:
    return RecommendationDTO(
        id=r.id,
        created_at=r.created_at.isoformat() if r.created_at else "",
        agent_type=r.agent_type,
        param_path=r.param_path,
        current_value=r.current_value,
        proposed_value=r.proposed_value,
        rationale=r.rationale,
        expected_effect=r.expected_effect,
        confidence=r.confidence,
        risk=r.risk,
        backtest_result=r.backtest_result,
        status=r.status,
        applied_at=r.applied_at.isoformat() if r.applied_at else None,
        rejected_reason=r.rejected_reason,
        notes=r.notes,
    )


@router.get("/")
async def list_recommendations(
    request: Request,
    status: str | None = None,
    limit: int = 50,
):
    """List recommendations, newest first. `status=pending` is the default
    operator-facing view; pass other statuses or None for full history."""
    from api.trades import _session_factory

    if not _session_factory:
        return []
    async with _session_factory() as session:
        stmt = select(AgentRecommendation)
        if status:
            stmt = stmt.where(AgentRecommendation.status == status)
        stmt = stmt.order_by(desc(AgentRecommendation.created_at)).limit(limit)
        rows = (await session.execute(stmt)).scalars().all()
        return [_to_dto(r) for r in rows]


@router.post("/{rec_id}/accept")
async def accept_recommendation(
    request: Request, rec_id: int, body: AcceptBody | None = None,
):
    """Apply the proposed yaml change + hot-reload + mark accepted.

    A4: yaml mutation goes through services.yaml_mutator (whitelist +
    type validation + .bak backup + atomic write). After the file is
    updated we trigger the same reload path /strategies/reload uses, so
    the live engines see the change without a backend restart.

    Failures (path not whitelisted, type mismatch, missing key) leave
    the recommendation as `pending` and surface a 422.
    """
    import logging
    from pathlib import Path

    from api.trades import _session_factory
    from services.yaml_mutator import YamlMutationError, apply_yaml_change

    logger = logging.getLogger(__name__)

    if not _session_factory:
        raise HTTPException(503, "DB unavailable")

    async with _session_factory() as session:
        rec = await session.get(AgentRecommendation, rec_id)
        if not rec:
            raise HTTPException(404, "recommendation not found")
        if rec.status != "pending":
            raise HTTPException(409, f"already {rec.status}")

        # Step 1: apply yaml change (synchronous file IO).
        yaml_path = Path(__file__).resolve().parent.parent.parent / "config" / "strategies.yaml"
        try:
            old_value, new_value = apply_yaml_change(
                yaml_path, rec.param_path, rec.proposed_value,
            )
        except YamlMutationError as e:
            logger.warning(
                "Recommendation #%d apply failed: %s", rec.id, e,
            )
            raise HTTPException(422, f"yaml apply failed: {e}") from e

        # Step 2: trigger hot-reload so the engines pick up the change.
        try:
            registry = getattr(request.app.state, "strategy_registry", None) \
                or getattr(request.app.state, "registry", None)
            if registry and hasattr(registry, "reload_config"):
                registry.reload_config()
            apply_kr = getattr(request.app.state, "apply_kr_eval_overrides", None)
            apply_us = getattr(request.app.state, "apply_us_eval_overrides", None)
            if apply_kr:
                apply_kr()
            if apply_us:
                apply_us()
            logger.info("Recommendation #%d reload triggered", rec.id)
        except Exception as e:
            # Yaml is already written. Log + continue — operator can
            # restart manually if reload glitches.
            logger.error(
                "Recommendation #%d hot-reload failed (yaml already applied): %s",
                rec.id, e,
            )

        # Step 3: mark accepted.
        rec.status = "accepted"
        rec.applied_at = now_utc_naive()
        if body and body.notes:
            rec.notes = body.notes
        await session.commit()
        logger.info(
            "Recommendation #%d accepted: %s : %r → %r",
            rec.id, rec.param_path, old_value, new_value,
        )
        return _to_dto(rec)


@router.post("/{rec_id}/reject")
async def reject_recommendation(
    request: Request, rec_id: int, body: RejectBody | None = None,
):
    from api.trades import _session_factory

    if not _session_factory:
        raise HTTPException(503, "DB unavailable")
    async with _session_factory() as session:
        rec = await session.get(AgentRecommendation, rec_id)
        if not rec:
            raise HTTPException(404, "recommendation not found")
        if rec.status != "pending":
            raise HTTPException(409, f"already {rec.status}")
        rec.status = "rejected"
        if body and body.reason:
            rec.rejected_reason = body.reason
        await session.commit()
        return _to_dto(rec)
