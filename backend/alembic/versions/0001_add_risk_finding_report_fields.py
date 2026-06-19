"""Add report fields to risk findings.

Revision ID: 0001_risk_fields
Revises:
Create Date: 2026-06-19
"""

from alembic import op

revision = "0001_risk_fields"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE risk_findings ADD COLUMN IF NOT EXISTS finding_category VARCHAR(64)")
    op.execute("ALTER TABLE risk_findings ADD COLUMN IF NOT EXISTS clause_group VARCHAR(64)")
    op.execute("ALTER TABLE risk_findings ADD COLUMN IF NOT EXISTS supporting_clauses_json JSON")
    op.execute("ALTER TABLE risk_findings ADD COLUMN IF NOT EXISTS negotiation_recommendation TEXT")


def downgrade():
    op.execute("ALTER TABLE risk_findings DROP COLUMN IF EXISTS negotiation_recommendation")
    op.execute("ALTER TABLE risk_findings DROP COLUMN IF EXISTS supporting_clauses_json")
    op.execute("ALTER TABLE risk_findings DROP COLUMN IF EXISTS clause_group")
    op.execute("ALTER TABLE risk_findings DROP COLUMN IF EXISTS finding_category")
