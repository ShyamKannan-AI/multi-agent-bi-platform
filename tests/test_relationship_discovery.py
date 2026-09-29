import pandas as pd
import pytest

from src.relationships import RelationshipDiscovery


def test_discovers_customer_order_relationship():

    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["A", "B", "C"],
    })

    orders = pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 1, 2, 3],
    })

    engine = RelationshipDiscovery()

    result = engine.discover({
        "customers": customers,
        "orders": orders,
    })

    assert result["status"] == "success"
    assert result["relationship_count"] >= 1

    matches = [
        relationship
        for relationship in result["relationships"]
        if (
            relationship["left_column"] == "customer_id"
            and relationship["right_column"] == "customer_id"
        )
        or (
            relationship["left_column"] == "customer_id"
            and relationship["right_column"] == "customer_id"
        )
    ]

    assert matches

    relationship = matches[0]

    assert relationship["confidence"] >= 0.60
    assert relationship["relationship"] in {
        "one_to_many",
        "many_to_one",
        "one_to_one",
        "reference",
        "possible_relationship",
    }


def test_detects_product_relationship():

    products = pd.DataFrame({
        "product_id": [10, 20, 30],
        "product": ["Laptop", "Mouse", "Keyboard"],
    })

    orders = pd.DataFrame({
        "order_id": [1, 2, 3],
        "product_id": [10, 10, 20],
    })

    engine = RelationshipDiscovery()

    result = engine.discover({
        "products": products,
        "orders": orders,
    })

    matches = [
        r
        for r in result["relationships"]
        if (
            r["left_column"] == "product_id"
            and r["right_column"] == "product_id"
        )
    ]

    assert matches
    assert matches[0]["confidence"] >= 0.60


def test_requires_multiple_datasets():

    engine = RelationshipDiscovery()

    df = pd.DataFrame({
        "id": [1, 2, 3],
    })

    with pytest.raises(ValueError):
        engine.discover({
            "data": df,
        })


def test_rejects_non_dataframe():

    engine = RelationshipDiscovery()

    with pytest.raises(TypeError):
        engine.discover({
            "customers": [1, 2, 3],
            "orders": pd.DataFrame({
                "id": [1, 2]
            }),
        })


def test_does_not_modify_dataframes():

    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
    })

    orders = pd.DataFrame({
        "customer_id": [1, 1, 2],
    })

    customers_before = customers.copy(deep=True)
    orders_before = orders.copy(deep=True)

    engine = RelationshipDiscovery()

    engine.discover({
        "customers": customers,
        "orders": orders,
    })

    pd.testing.assert_frame_equal(
        customers,
        customers_before,
    )

    pd.testing.assert_frame_equal(
        orders,
        orders_before,
    )