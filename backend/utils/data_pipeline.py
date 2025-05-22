import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from balldontlie import BalldontlieAPI
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Initialize balldontlie client
api = BalldontlieAPI(api_key="bfdc4ecf-c070-4e93-b9ac-cb36f049efb1")

async def get_future_games():
    """Get list of future games from balldontlie API."""
    try:
        # Get today's date and next 7 days
        today = datetime.now()
        dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        
        logger.info(f"Fetching future games for dates: {dates}")
        
        # Get games using the API
        response = api.nba.games.list(dates=dates)
        games = response.data
        
        if not games:
            logger.info(f"No future games found")
            return []
            
        # Format games for scraping
        future_games = []
        for game in games:
            if game.status != 'Final':  # Only get future games
                future_games.append({
                    'home_team': game.home_team.abbreviation,
                    'away_team': game.visitor_team.abbreviation,
                    'date': game.date
                })
                
        return future_games
        
    except Exception as e:
        logger.error(f"Error fetching future games: {str(e)}")
        return []

async def scrape_team_stats(team_abbr, date):
    """Scrape team stats from basketball-reference.com."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Format date for URL
            year = date.split('-')[0]
            month = date.split('-')[1]
            day = date.split('-')[2]
            
            # Construct URL for team stats
            url = f"https://www.basketball-reference.com/teams/{team_abbr}/{year}.html"
            
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            
            # Get team stats
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract required stats
            stats = {}
            
            # Team shooting stats
            shooting_table = soup.find('table', {'id': 'team_shooting'})
            if shooting_table:
                stats['fg%'] = float(shooting_table.find('td', {'data-stat': 'fg_pct'}).text)
                stats['3p'] = float(shooting_table.find('td', {'data-stat': 'fg3'}).text)
                stats['efg%'] = float(shooting_table.find('td', {'data-stat': 'efg_pct'}).text)
            
            # Team advanced stats
            advanced_table = soup.find('table', {'id': 'team_misc'})
            if advanced_table:
                stats['ast%'] = float(advanced_table.find('td', {'data-stat': 'ast_pct'}).text)
                stats['usg%'] = float(advanced_table.find('td', {'data-stat': 'usg_pct'}).text)
                stats['ortg'] = float(advanced_table.find('td', {'data-stat': 'off_rtg'}).text)
                stats['drtg'] = float(advanced_table.find('td', {'data-stat': 'def_rtg'}).text)
            
            # Team per game stats
            per_game_table = soup.find('table', {'id': 'team_and_opponent'})
            if per_game_table:
                stats['trb'] = float(per_game_table.find('td', {'data-stat': 'trb_per_g'}).text)
                stats['fga_max'] = float(per_game_table.find('td', {'data-stat': 'fga_per_g'}).text)
                stats['3pa_max'] = float(per_game_table.find('td', {'data-stat': 'fg3a_per_g'}).text)
                stats['ft_max'] = float(per_game_table.find('td', {'data-stat': 'ft_per_g'}).text)
                stats['orb_max'] = float(per_game_table.find('td', {'data-stat': 'orb_per_g'}).text)
                stats['stl_max'] = float(per_game_table.find('td', {'data-stat': 'stl_per_g'}).text)
                stats['tov_max'] = float(per_game_table.find('td', {'data-stat': 'tov_per_g'}).text)
                stats['blk_max'] = float(per_game_table.find('td', {'data-stat': 'blk_per_g'}).text)
            
            await browser.close()
            return stats
            
    except Exception as e:
        logger.error(f"Error scraping stats for {team_abbr}: {str(e)}")
        return None

async def process_future_games():
    """Process future games by scraping detailed stats."""
    try:
        # Get future games
        future_games = await get_future_games()
        if not future_games:
            return None
            
        # Process each game
        processed_games = []
        for game in future_games:
            try:
                # Scrape stats for both teams
                home_stats = await scrape_team_stats(game['home_team'], game['date'])
                away_stats = await scrape_team_stats(game['away_team'], game['date'])
                
                if not home_stats or not away_stats:
                    continue
                    
                # Create game entry
                game_entry = {
                    'team': game['home_team'],
                    'team_opp': game['away_team'],
                    'date': game['date'],
                    'home': 1,
                    'won': None,  # Future game
                    'total': None,  # Future game
                    'total_opp': None,  # Future game
                    'fg%': home_stats['fg%'],
                    '3p': home_stats['3p'],
                    'trb': home_stats['trb'],
                    'efg%': home_stats['efg%'],
                    'ast%': home_stats['ast%'],
                    'usg%': home_stats['usg%'],
                    'ortg': home_stats['ortg'],
                    'fga_max': home_stats['fga_max'],
                    '3pa_max': home_stats['3pa_max'],
                    'ft_max': home_stats['ft_max'],
                    'orb_max': home_stats['orb_max'],
                    'gmsc_max': None,  # Not available for future games
                    'ftr_max': None,  # Not available for future games
                    'stl%_max': None,  # Not available for future games
                    'blk%_max': None,  # Not available for future games
                    'fg%_opp': away_stats['fg%'],
                    'ast_opp': None,  # Not available for future games
                    'pts_opp': None,  # Not available for future games
                    'ts%_opp': None,  # Not available for future games
                    'efg%_opp': away_stats['efg%'],
                    'blk%_opp': None,  # Not available for future games
                    'usg%_opp': away_stats['usg%'],
                    'drtg_opp': away_stats['drtg'],
                    'fg%_max_opp': away_stats['fg%'],
                    'stl_max_opp': away_stats['stl_max'],
                    'tov_max_opp': away_stats['tov_max'],
                    'gmsc_max_opp': None,  # Not available for future games
                    'drb%_max_opp': None,  # Not available for future games
                    'ast%_max_opp': away_stats['ast%'],
                    'total_opp': None  # Future game
                }
                processed_games.append(game_entry)
                
            except Exception as e:
                logger.error(f"Error processing game: {str(e)}")
                continue
                
        if not processed_games:
            logger.warning("No valid games to save after processing")
            return None
            
        # Create DataFrame for future games
        future_games_df = pd.DataFrame(processed_games)
        return future_games_df
        
    except Exception as e:
        logger.error(f"Error processing future games: {str(e)}")
        return None

async def update_data():
    """Update the NBA game data by processing future games."""
    try:
        # Create processed directory if it doesn't exist
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        
        # Process future games
        logger.info("Processing future games...")
        games_df = await process_future_games()
        
        if games_df is None:
            raise ValueError("No games data available")
        
        # Save the dataset
        output_path = os.path.join(PROCESSED_DIR, "final_dataset.csv")
        games_df.to_csv(output_path, index=False)
        logger.info(f"Data update complete. Saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        raise 