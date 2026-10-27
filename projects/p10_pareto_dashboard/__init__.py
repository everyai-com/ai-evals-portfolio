"""p10 Cost-Quality Pareto Dashboard: routing-policy frontier per tenant, with price sensitivity."""
from projects.p10_pareto_dashboard.core import (
    PolicyStats,
    aggregate,
    build_metrics,
    marginal_costs,
    pareto_frontier,
    price_sensitivity,
)

__all__ = ["PolicyStats", "aggregate", "build_metrics", "marginal_costs", "pareto_frontier", "price_sensitivity"]
