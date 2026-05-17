from google_play_scraper import reviews, Sort
import pandas as pd


BANK_APPS = {
    "CBE": "com.combanketh.mobilebanking",
    "BOA": "com.boa.boaMobileBanking",
    "Dashen": "com.dashen.dashensuperapp"
}


def scrape_bank_reviews(app_id, bank_name, count=500):
    """
    Scrape reviews from Google Play Store.
    """

    result, _ = reviews(
        app_id,
        lang="en",
        country="et",
        sort=Sort.NEWEST,
        count=count
    )

    reviews_data = []

    for review in result:
        reviews_data.append({
            "review": review["content"],
            "rating": review["score"],
            "date": review["at"],
            "bank": bank_name,
            "source": "Google Play"
        })

    return pd.DataFrame(reviews_data)


def main():
    all_reviews = []

    for bank, app_id in BANK_APPS.items():
        print(f"Scraping reviews for {bank}...")

        df = scrape_bank_reviews(app_id, bank)

        print(f"{bank}: {len(df)} reviews collected")

        all_reviews.append(df)

    combined_df = pd.concat(all_reviews, ignore_index=True)

    combined_df.to_csv(
        "data/raw/bank_reviews_raw.csv",
        index=False
    )

    print("Raw reviews saved successfully.")


if __name__ == "__main__":
    main()