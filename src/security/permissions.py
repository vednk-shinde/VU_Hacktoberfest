from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class UserContext:
    """
    Encapsulates user authentication, tenant identity, and dataset/column permissions.
    """
    user_id: str
    roles: List[str] = field(default_factory=lambda: ["viewer"])
    allowed_datasets: List[str] = field(default_factory=list)  # Explicit grants are required.
    allowed_columns: Dict[str, List[str]] = field(default_factory=dict)  # dataset_id -> list of cols


def can_access_dataset(user: Optional[UserContext], dataset_id: str) -> bool:
    """Verifies whether the user is permitted to query the specified dataset."""
    if user is None:
        return False  # Registered datasets require an explicit authorization context.
    if "*" in user.allowed_datasets or dataset_id in user.allowed_datasets:
        return True
    return False


def can_access_columns(
    user: Optional[UserContext],
    dataset_id: str,
    columns: List[str],
) -> Tuple[bool, List[str]]:
    """
    Verifies column-level permissions.
    Returns (is_permitted, list_of_unauthorized_columns).
    """
    if user is None:
        return False, list(columns)

    permitted = user.allowed_columns.get(dataset_id)
    if permitted is None or "*" in permitted:
        return True, []

    permitted_set = set(c.lower() for c in permitted)
    unauthorized = [c for c in columns if c.lower() not in permitted_set]

    return len(unauthorized) == 0, unauthorized
