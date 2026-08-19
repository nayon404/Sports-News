import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# ---------------------------------------
# SETTINGS
# ---------------------------------------

TARGET_DATA = 1800

BASE_URL = "https://www.thedailystar.net/sports"

# Store all news
news_data = []

# Store URLs to avoid duplicates
used_urls = set()

# Browser information
headers = {
    "User-Agent": "Mozilla/5.0"
}


# ---------------------------------------
# START SCRAPING
# ---------------------------------------

page = 1

while len(news_data) < TARGET_DATA:

    # Create page URL
    if page == 1:
        url = BASE_URL
    else:
        url = BASE_URL + "?page=" + str(page)

    print("Scraping page:", page)

    # Send request
    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    # Check response
    if response.status_code != 200:
        print("Page could not be opened.")
        page += 1
        continue

    # Create BeautifulSoup
    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # ---------------------------------------
    # FIND NEWS ARTICLES
    # ---------------------------------------

    # Find all links
    links = soup.find_all("a", href=True)

    page_count = 0

    for link in links:

        title = link.get_text(
            " ",
            strip=True
        )

        href = link["href"]

        # ---------------------------------------
        # CHECK SPORTS ARTICLE
        # ---------------------------------------

        if "/sports/" not in href:
            continue

        # Ignore empty titles
        if title == "":
            continue

        # Create full URL
        if href.startswith("http"):
            article_url = href
        else:
            article_url = "https://www.thedailystar.net" + href

        # Remove duplicate URLs
        if article_url in used_urls:
            continue

        # ---------------------------------------
        # FIND DATE
        # ---------------------------------------

        date = ""

        # Look for time tag
        time_tag = link.find("time")

        if time_tag:

            date = time_tag.get_text(
                " ",
                strip=True
            )

        # If date was not inside <time>,
        # look at the parent section
        if date == "":

            parent = link.parent

            if parent:

                parent_text = parent.get_text(
                    " ",
                    strip=True
                )

                # Check for common date pattern
                if "2026" in parent_text:

                    date = parent_text


        # ---------------------------------------
        # FIND CATEGORY
        # ---------------------------------------

        category = "Sports"

        href_lower = href.lower()

        if "/cricket/" in href_lower:
            category = "Cricket"

        elif "/football/" in href_lower:
            category = "Football"

        elif "/tennis/" in href_lower:
            category = "Tennis"

        elif "/more-sports/" in href_lower:
            category = "More Sports"

        elif "/sports-special/" in href_lower:
            category = "Sports Special"


        # ---------------------------------------
        # SAVE DATA
        # ---------------------------------------

        used_urls.add(article_url)

        news_data.append({
            "Title": title,
            "Category": category,
            "Date": date,
            "URL": article_url
        })

        page_count += 1

        # Stop at 1800
        if len(news_data) >= TARGET_DATA:
            break


    # ---------------------------------------
    # PAGE INFORMATION
    # ---------------------------------------

    print(
        "News found:",
        page_count
    )

    print(
        "Total collected:",
        len(news_data)
    )

    print("-" * 50)

    # Next page
    page += 1

    # Wait 1 second
    time.sleep(1)


# ---------------------------------------
# CREATE DATAFRAME
# ---------------------------------------

print("\nCreating DataFrame...")

df = pd.DataFrame(news_data)


# ---------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------

df = df.drop_duplicates(
    subset=["URL"]
)


# ---------------------------------------
# KEEP 1800 RECORDS
# ---------------------------------------

df = df.head(1800)


# ---------------------------------------
# SAVE CSV
# ---------------------------------------

df.to_csv(
    "daily_star_sports_news.csv",
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------
# SHOW RESULT
# ---------------------------------------

print("\n" + "=" * 50)
print("SCRAPING COMPLETED")
print("=" * 50)

print(
    "Total records:",
    len(df)
)

print(
    "CSV file: daily_star_sports_news.csv"
)

print("=" * 50)