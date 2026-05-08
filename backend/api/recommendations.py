"""Self-evolution recommendations API.

Surfaces AgentRecommendation rows so the operator can accept/reject
proposed config changes from agents (trade_review for now).
"""

from datetime import datetime
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
    """Mark accepted. Yaml hot-update is wired in track A4; for now this
    just records the decision so the operator can apply manually."""
    from api.trades import _session_factory

    if not _session_factory:
        raise HTTPException(503, "DB unavailable")
    async with _session_factory() as session:
        rec = await session.get(AgentRecommendation, rec_id)
        if not rec:
            raise HTTPException(404, "recommendation not found")
        if rec.status != "pending":
            raise HTTPException(409, f"already {rec.status}")
        rec.status = "accepted"
        rec.applied_at = datetime.utcnow()
        if body and body.notes:
            rec.notes = body.notes
        await session.commit()
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
