from data_preprocessing import get_past_matches, calculate_team_stats
from feature_engineering import head_to_head_stats
from team_mapping import TEAM_NAME_MAPPING
from data_preprocessing import get_recent_form
import pandas as pd

def extract_features(df, home_team, away_team, match_date, le_team):
    home_team = TEAM_NAME_MAPPING.get(home_team.upper(), home_team.upper())
    away_team = TEAM_NAME_MAPPING.get(away_team.upper(), away_team.upper())

    past_home = get_past_matches(df, home_team, match_date)
    home_stats = calculate_team_stats(past_home, home_team)

    past_away = get_past_matches(df, away_team, match_date)
    away_stats = calculate_team_stats(past_away, away_team)

    h2h = head_to_head_stats(df, home_team, away_team, match_date)
    
    recent_home = get_recent_form(df, home_team, match_date)
    recent_away = get_recent_form(df, away_team, match_date)

    home_team_enc = le_team.transform([home_team])[0]
    away_team_enc = le_team.transform([away_team])[0]

    return pd.DataFrame([{
        'home_team_enc': home_team_enc,
        'away_team_enc': away_team_enc,
        'home_wins_last3y': home_stats['wins'],
        'home_draws_last3y': home_stats['draws'],
        'home_losses_last3y': home_stats['losses'],
        'home_goals_for_last3y': home_stats['goals_for'],
        'home_goals_against_last3y': home_stats['goals_against'],
        'away_wins_last3y': away_stats['wins'],
        'away_draws_last3y': away_stats['draws'],
        'away_losses_last3y': away_stats['losses'],
        'away_goals_for_last3y': away_stats['goals_for'],
        'away_goals_against_last3y': away_stats['goals_against'],
        'home_h2h_winrate': h2h,
        
        'recent_home_wins': recent_home['recent_wins'],
        'recent_home_draws': recent_home['recent_draws'],
        'recent_home_losses': recent_home['recent_losses'],
        'recent_home_goals_for': recent_home['recent_goals_for'],
        'recent_home_goals_against': recent_home['recent_goals_against'],

        'recent_away_wins': recent_away['recent_wins'],
        'recent_away_draws': recent_away['recent_draws'],
        'recent_away_losses': recent_away['recent_losses'],
        'recent_away_goals_for': recent_away['recent_goals_for'],
        'recent_away_goals_against': recent_away['recent_goals_against'],
    }])