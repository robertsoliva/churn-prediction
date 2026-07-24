"""Download the Telco Customer Churn dataset from IBM's public repository."""
import os
import requests

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
DEST = os.path.join("data", "telco_churn.csv")


def download() -> None:
    os.makedirs("data", exist_ok=True)
    print(f"Downloading from {URL} …")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    with open(DEST, "wb") as f:
        f.write(r.content)
    print(f"Saved to {DEST}  ({len(r.content) / 1024:.0f} KB)")


if __name__ == "__main__":
    download()
