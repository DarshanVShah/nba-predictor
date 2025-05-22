import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBAScraper:
    def __init__(self):
        self.base_url = "https://www.basketball-reference.com"
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)

    async def get_html(self, url, selector, sleep=5, retries=3):
        """Get HTML content from a URL using Playwright"""
        html = None
        for i in range(1, retries+1):
            time.sleep(sleep * i)
            try:
                async with async_playwright() as p:
                    browser = await p.firefox.launch()
                    page = await browser.new_page()
                    await page.goto(url)
                    html = await page.inner_html(selector)
                    await browser.close()
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                continue
            else:
                break
        return html

    async def get_team_stats(self, team_abbr):
        """Get team's recent performance stats"""
        try:
            # Get team's page
            url = f"{self.base_url}/teams/{team_abbr}/2024.html"
            html = await self.get_html(url, "#team_and_opponent")
            if not html:
                return None

            soup = BeautifulSoup(html, 'html.parser')
            
            # Get last 10 games record
            last_10 = soup.find('div', {'data-template': 'Partials/Teams/Summary'})
            if last_10:
                last_10_text = last_10.find('p', string=lambda x: x and 'Last 10' in x)
                if last_10_text:
                    wins = int(last_10_text.text.split('(')[1].split('-')[0])
                    win_pct_last_10 = wins / 10
                else:
                    win_pct_last_10 = 0.5
            else:
                win_pct_last_10 = 0.5

            # Get season record
            season_record = soup.find('div', {'data-template': 'Partials/Teams/Summary'})
            if season_record:
                record_text = season_record.find('p', string=lambda x: x and 'Record:' in x)
                if record_text:
                    wins = int(record_text.text.split('(')[1].split('-')[0])
                    total = int(record_text.text.split('(')[1].split('-')[1].split(')')[0])
                    win_pct_season = wins / total if total > 0 else 0.5
                else:
                    win_pct_season = 0.5
            else:
                win_pct_season = 0.5

            # Calculate momentum score (weighted average of last 10 games)
            momentum_score = (win_pct_last_10 * 0.7) + (win_pct_season * 0.3)

            return {
                'win_pct_last_10': win_pct_last_10,
                'win_pct_season': win_pct_season,
                'momentum_score': momentum_score
            }
        except Exception as e:
            logger.error(f"Error getting stats for {team_abbr}: {str(e)}")
            return None

    async def get_opponent_strength(self, team_abbr):
        """Calculate opponent strength based on their season performance"""
        try:
            url = f"{self.base_url}/teams/{team_abbr}/2024.html"
            html = await self.get_html(url, "#team_and_opponent")
            if not html:
                return 0.5

            soup = BeautifulSoup(html, 'html.parser')
            
            # Get opponent's points per game
            opp_stats = soup.find('div', {'data-template': 'Partials/Teams/Summary'})
            if opp_stats:
                opp_ppg = opp_stats.find('p', string=lambda x: x and 'Opp PTS/G' in x)
                if opp_ppg:
                    opp_strength = float(opp_ppg.text.split(':')[1].strip())
                    # Normalize to 0-1 scale (assuming typical NBA scores)
                    opp_strength = min(max((opp_strength - 100) / 20, 0), 1)
                    return opp_strength

            return 0.5
        except Exception as e:
            logger.error(f"Error getting opponent strength for {team_abbr}: {str(e)}")
            return 0.5

    async def process_game(self, game_data):
        """Process a single game's data"""
        try:
            home_team = game_data['home_team']['abbreviation']
            away_team = game_data['visitor_team']['abbreviation']
            game_date = game_data['date']

            # Get team stats
            home_stats = await self.get_team_stats(home_team)
            away_stats = await self.get_team_stats(away_team)

            if not home_stats or not away_stats:
                logger.warning(f"Could not get stats for {home_team} vs {away_team}")
                return None

            # Get opponent strength
            home_opp_strength = await self.get_opponent_strength(home_team)
            away_opp_strength = await self.get_opponent_strength(away_team)

            # Create game entry
            game_entry = {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date,
                'home_court_advantage': 1,
                'rest_days': 0,  # Could be calculated from schedule
                'win_pct_last_10': home_stats['win_pct_last_10'],
                'win_pct_season': home_stats['win_pct_season'],
                'momentum_score': home_stats['momentum_score'],
                'opp_strength': home_opp_strength
            }

            return game_entry
        except Exception as e:
            logger.error(f"Error processing game: {str(e)}")
            return None

    async def process_daily_games(self, games):
        """Process all daily games and append to final_dataset.csv"""
        try:
            processed_games = []
            for game in games:
                game_entry = await self.process_game(game)
                if game_entry:
                    processed_games.append(game_entry)

            if not processed_games:
                logger.warning("No valid games to save after processing")
                return None

            # Create DataFrame for new games
            new_games_df = pd.DataFrame(processed_games)
            
            # Load existing final_dataset.csv if it exists
            final_dataset_path = os.path.join(self.data_dir, "final_dataset.csv")
            if os.path.exists(final_dataset_path):
                existing_df = pd.read_csv(final_dataset_path)
                # Append new games to existing dataset
                final_df = pd.concat([existing_df, new_games_df], ignore_index=True)
            else:
                final_df = new_games_df

            # Save updated dataset
            final_df.to_csv(final_dataset_path, index=False)
            logger.info(f"Added {len(processed_games)} new games to final_dataset.csv")

            # Also save to daily_games.csv for reference
            daily_games_path = os.path.join(self.data_dir, "daily_games.csv")
            new_games_df.to_csv(daily_games_path, index=False)
            logger.info(f"Saved {len(processed_games)} games to daily_games.csv")

            return new_games_df
        except Exception as e:
            logger.error(f"Error processing daily games: {str(e)}")
            return None

async def main():
    """Main function to test the scraper"""
    scraper = NBAScraper()
    
    # Test with a sample game
    test_game = {
        'home_team': {'abbreviation': 'NYK'},
        'visitor_team': {'abbreviation': 'IND'},
        'date': datetime.now().strftime("%Y-%m-%d")
    }
    
    result = await scraper.process_game(test_game)
    if result:
        print("Successfully processed test game:")
        print(result)
    else:
        print("Failed to process test game")

if __name__ == "__main__":
    asyncio.run(main()) 