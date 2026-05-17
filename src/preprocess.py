import pandas as pd


def preprocess_reviews(input_path, output_path):
    """
    Clean and preprocess scraped reviews.
    """

    df = pd.read_csv(input_path)

    print("Initial Shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    print("After Removing Duplicates:", df.shape)

    # Remove missing review text or rating
    missing_before = df.isnull().sum()

    print("Missing Values Before:")
    print(missing_before)

    df = df.dropna(subset=["review", "rating"])

    missing_after = df.isnull().sum()

    print("Missing Values After:")
    print(missing_after)

    # Normalize dates
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Ensure correct column order
    df = df[[
        "review",
        "rating",
        "date",
        "bank",
        "source"
    ]]

    # Save cleaned data
    df.to_csv(output_path, index=False)

    print(f"Cleaned dataset saved to {output_path}")

    print("Final Shape:", df.shape)

    return df


if __name__ == "__main__":
    preprocess_reviews(
        "data/raw/bank_reviews_raw.csv",
        "data/raw/bank_reviews_cleaned.csv"
    )