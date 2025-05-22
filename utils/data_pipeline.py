import os
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "NBAdata"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")
SEASONS = list(range(2020, 2026))

async def get_html(url, selector, sleep=5, retries=3):
    """Get HTML content from a URL using Playwright."""
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep * i)
        
        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                logger.info(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            logger.warning(f"Timeout error on {url}")
            continue
        else:
            break
            
    return html

async def scrape_season(season):
    """Scrape all games for a given season."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    
    if not html:
        logger.error(f"Failed to get HTML for season {season}")
        return
    
    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    standings_pages = [f"https://basketball-reference.com{l['href']}" for l in links]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue
            
        html = await get_html(url, "#all_schedule")
        if not html:
            continue
            
        with open(save_path, "w+") as f:
            f.write(html)

async def scrape_game(standings_file):
    """Scrape individual game data from a standings file."""
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in box_scores]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#content")
        if not html:
            continue
            
        with open(save_path, "w+") as f:
            f.write(html)

def parse_html(box_score):
    """Parse HTML content from a box score file."""
    with open(box_score) as f:
        html = f.read()
        
    soup = BeautifulSoup(html)
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_line_score(soup):
    """Extract line score data from parsed HTML."""
    line_score = pd.read_html(str(soup), attrs={"id": "line_score"})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    
    line_score = line_score[["team", "total"]]
    return line_score

def read_stats(soup, team, stat):
    """Extract team statistics from parsed HTML."""
    df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def read_season_info(soup):
    """Extract season information from parsed HTML."""
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

def process_games():
    """Process all scraped game data into a single DataFrame."""
    box_scores = [os.path.join(SCORES_DIR, f) for f in os.listdir(SCORES_DIR) if f.endswith(".html")]
    
    base_cols = None
    games = []

    for idx, box_score in enumerate(box_scores):
        try:
            soup = parse_html(box_score)
        except Exception as e:
            logger.error(f"Error parsing {box_score} at index {idx}: {e}")
            continue
        
        try:
            line_score = read_line_score(soup)
            teams = list(line_score["team"])

            summaries = []
            for team in teams:
                basic = read_stats(soup, team, "basic")
                advanced = read_stats(soup, team, "advanced")

                totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])
                totals.index = totals.index.str.lower()

                maxes = pd.concat([basic.iloc[:-1,:].max(), advanced.iloc[:-1,:].max()])
                maxes.index = maxes.index.str.lower() + "_max"

                summary = pd.concat([totals, maxes])

                if base_cols is None:
                    base_cols = list(summary.index.drop_duplicates(keep="first"))
                    base_cols = [b for b in base_cols if "bpm" not in b]

                summary = summary[base_cols]
                summaries.append(summary)
                
            summary = pd.concat(summaries, axis=1).T
            game = pd.concat([summary, line_score], axis=1)

            game["home"] = [0,1]
            game_opp = game.iloc[::-1].reset_index()
            game_opp.columns += "_opp"

            full_game = pd.concat([game, game_opp], axis=1)
            full_game["season"] = read_season_info(soup)
            full_game["date"] = os.path.basename(box_score)[:8]
            full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")
            full_game["won"] = full_game["total"] > full_game["total_opp"]
            
            games.append(full_game)

            if len(games) % 100 == 0:
                logger.info(f"Processed {len(games)} / {len(box_scores)} games")

        except Exception as e:
            logger.error(f"Error processing game from {box_score} at index {idx}: {e}")
            continue

    if not games:
        logger.error("No games were successfully processed")
        return None

    games_df = pd.concat(games, ignore_index=True)
    return games_df

async def update_data():
    """Update the NBA game data by scraping and processing new games."""
    # Create directories if they don't exist
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    os.makedirs(SCORES_DIR, exist_ok=True)
    
    # Scrape data for each season
    for season in SEASONS:
        logger.info(f"Scraping season {season}")
        await scrape_season(season)
    
    # Scrape individual games
    standings_files = [os.path.join(STANDINGS_DIR, f) for f in os.listdir(STANDINGS_DIR) if f.endswith(".html")]
    for standings_file in standings_files:
        logger.info(f"Scraping games from {standings_file}")
        await scrape_game(standings_file)
    
    # Process the data
    logger.info("Processing game data")
    games_df = process_games()
    
    if games_df is not None:
        games_df.to_csv("nba_games.csv", index=False)
        logger.info("Data update complete")
    else:
        logger.error("Data update failed") 