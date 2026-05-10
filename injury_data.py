import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_injury_report():
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        injuries = []
        tables = soup.find_all('div', class_='TableBaseWrapper')
        
        for table in tables:
            team_header = table.find('h4')
            if not team_header:
                continue
            team_name = team_header.text.strip()
            
            rows = table.find_all('tr')[1:]  # skip header
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    player = cols[0].text.strip()
                    position = cols[1].text.strip()
                    status = cols[2].text.strip()
                    injuries.append({
                        'team': team_name,
                        'player': player,
                        'position': position,
                        'status': status
                    })
        
        return pd.DataFrame(injuries)
    
    except Exception as e:
        print("Error fetching injuries: " + str(e))
        return pd.DataFrame()

def is_star_injured(team_abbr, star_name, injury_df):
    if injury_df.empty:
        return False
    # Check if star player appears in injury report as Out
    for _, row in injury_df.iterrows():
        if star_name.lower() in row['player'].lower():
            if 'out' in row['status'].lower():
                return True
    return False

def get_team_injuries(team_name_partial, injury_df):
    if injury_df.empty:
        return []
    matches = injury_df[injury_df['team'].str.contains(team_name_partial, case=False)]
    return matches.to_dict('records')

if __name__ == "__main__":
    print("Fetching injury report...")
    df = get_injury_report()
    if not df.empty:
        print("Found " + str(len(df)) + " injured players")
        print(df.head(10).to_string())
    else:
        print("No injury data found")