def estimate_price(sqft, beds):
    print(f"🏠 Estimating Valuation for {sqft} sqft, {beds} beds...")
    return sqft * 320 + beds * 15000

def main():
    val = estimate_price(1850, 3)
    print(f"✓ Estimated Market Value: ${val:,.2f}")

if __name__ == "__main__":
    main()
