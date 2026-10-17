from app.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank

RELEVANT = {("a.pdf", 1), ("a.pdf", 2)}


def test_recall_at_k():
    retrieved = [("a.pdf", 1), ("b.pdf", 1), ("a.pdf", 2), ("b.pdf", 2)]
    assert recall_at_k(retrieved, RELEVANT, k=1) == 0.5
    assert recall_at_k(retrieved, RELEVANT, k=3) == 1.0
    assert recall_at_k(retrieved, RELEVANT, k=10) == 1.0


def test_recall_with_no_relevant_set_is_zero():
    assert recall_at_k([("a.pdf", 1)], set(), k=1) == 0.0


def test_precision_at_k():
    retrieved = [("a.pdf", 1), ("b.pdf", 1), ("a.pdf", 2), ("b.pdf", 2)]
    assert precision_at_k(retrieved, RELEVANT, k=1) == 1.0
    assert precision_at_k(retrieved, RELEVANT, k=2) == 0.5
    assert precision_at_k(retrieved, RELEVANT, k=4) == 0.5


def test_precision_with_invalid_k():
    assert precision_at_k([("a.pdf", 1)], RELEVANT, k=0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([("b.pdf", 1), ("a.pdf", 1)], RELEVANT) == 0.5
    assert reciprocal_rank([("a.pdf", 1), ("b.pdf", 1)], RELEVANT) == 1.0
    assert reciprocal_rank([("b.pdf", 1), ("c.pdf", 1)], RELEVANT) == 0.0
