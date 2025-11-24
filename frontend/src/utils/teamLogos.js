/**
 * Team Logo Mapping Utility
 * Maps NBA team abbreviations to local logo assets
 */

// Map team abbreviations to local logo file names
const TEAM_LOGO_MAP = {
  'ATL': 'atl.png',
  'BOS': 'bos.png',
  'BRK': 'nets.png',  // Brooklyn Nets
  'BKN': 'nets.png',  // Brooklyn Nets (alternate abbreviation)
  'CHA': 'cha.png',   // Charlotte Hornets
  'CHO': 'cha.png',   // Charlotte Hornets (alternate abbreviation)
  'CHI': 'chi.png',
  'CLE': 'cav.png',   // Cleveland Cavaliers
  'DAL': 'dal.png',
  'DEN': 'nug.png',   // Denver Nuggets
  'DET': 'det.png',
  'GSW': 'gst.png',   // Golden State Warriors
  'HOU': 'hou.png',
  'IND': 'pac.png',   // Indiana Pacers
  'LAC': 'lac.png',   // LA Clippers
  'LAL': 'lal.png',   // LA Lakers
  'MEM': 'mem.png',   // Memphis Grizzlies
  'MIA': 'mia.gif',   // Miami Heat (note: it's a gif)
  'MIL': 'mil.png',   // Milwaukee Bucks
  'MIN': 'min.png',   // Minnesota Timberwolves
  'NOP': 'nor.png',   // New Orleans Pelicans
  'NYK': 'nyk.png',   // New York Knicks
  'OKC': 'okc.png',   // Oklahoma City Thunder
  'ORL': 'orl.png',   // Orlando Magic
  'PHI': 'phi.png',   // Philadelphia 76ers
  'PHX': 'sun.png',   // Phoenix Suns
  'PHO': 'sun.png',   // Phoenix Suns (alternate abbreviation)
  'POR': 'por.png',   // Portland Trail Blazers
  'SAC': 'sac.png',   // Sacramento Kings
  'SAS': 'san.png',   // San Antonio Spurs
  'TOR': 'tor.png',   // Toronto Raptors
  'UTA': 'uta.png',   // Utah Jazz
  'WAS': 'wiz.png',   // Washington Wizards
};

/**
 * Get the local logo path for a team abbreviation
 * @param {string} teamAbbr - Team abbreviation (e.g., 'LAL', 'BOS')
 * @returns {string} - Path to the logo file
 */
export const getTeamLogo = (teamAbbr) => {
  if (!teamAbbr) return null;
  
  const logoFile = TEAM_LOGO_MAP[teamAbbr.toUpperCase()];
  if (!logoFile) {
    console.warn(`No logo found for team: ${teamAbbr}`);
    return null;
  }
  
  // Return path relative to public folder (logos will be copied there)
  return `/assets/${logoFile}`;
};

/**
 * Get full team name from abbreviation
 * @param {string} teamAbbr - Team abbreviation
 * @returns {string} - Full team name
 */
export const getTeamName = (teamAbbr) => {
  const teamNames = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'BRK': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHO': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'PHO': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
  };
  
  return teamNames[teamAbbr?.toUpperCase()] || teamAbbr;
};

/**
 * Get short team name (without city) from abbreviation
 * @param {string} teamAbbr - Team abbreviation
 * @returns {string} - Short team name
 */
export const getShortTeamName = (teamAbbr) => {
  const fullName = getTeamName(teamAbbr);
  // Extract the team name part (after the city)
  const parts = fullName.split(' ');
  if (parts.length > 2) {
    return parts.slice(1).join(' ');
  }
  return fullName;
};

