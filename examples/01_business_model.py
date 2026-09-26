"""Example 1: A simple mechanistic business model."""

def business_model(
    traffic,
    conversion_rate,
    average_order_value,
    costs
):
    customers = traffic * conversion_rate
    revenue = customers * average_order_value
    profit = revenue - costs

    return {
        "customers": customers,
        "revenue": revenue,
        "profit": profit,
    }


if __name__ == "__main__":
    result = business_model(
        traffic=100_000,
        conversion_rate=0.03,
        average_order_value=75,
        costs=100_000,
    )

    for name, value in result.items():
        print(f"{name}: {value:,.2f}")
