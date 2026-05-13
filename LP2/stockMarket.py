# Expert System for Stock Market Trading
# Python Program

def stock_market_expert_system():

    print("======================================")
    print(" Stock Market Trading Expert System ")
    print("======================================")

    while True:

        print("\nEnter Market Details")

        risk = input("Risk Level (low/medium/high): ").lower()
        investment = int(input("Investment Amount (in ₹): "))
        market_trend = input("Market Trend (bullish/bearish/stable): ").lower()

        print("\nAnalyzing Market Conditions...\n")

        # Rule-Based Expert System
        if risk == "low" and market_trend == "stable":

            advice = "Invest in Blue-chip stocks or Mutual Funds."

        elif risk == "medium" and market_trend == "bullish":

            advice = "Consider investing in growth stocks."

        elif risk == "high" and market_trend == "bullish":

            advice = "High-risk trading like intraday or crypto can be considered."

        elif market_trend == "bearish":

            advice = "Avoid risky investments and focus on defensive stocks."

        else:

            advice = "Diversify your portfolio and monitor the market regularly."

        # Display Result
        print("===================================")
        print(" Trading Recommendation Report ")
        print("===================================")

        print(f"Risk Level       : {risk}")
        print(f"Investment Amount: ₹{investment}")
        print(f"Market Trend     : {market_trend}")

        print(f"\nExpert Advice    : {advice}")

        # Continue option
        again = input("\nDo you want another recommendation? (yes/no): ").lower()

        if again != "yes":
            print("\nExiting Expert System...")
            print("Thank You!")
            break


# Run Expert System
stock_market_expert_system()