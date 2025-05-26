from data_preprocessing import get_past_matches, calculate_team_stats
from feature_engineering import head_to_head_stats
from team_mapping import TEAM_NAME_MAPPING
from data_preprocessing import get_recent_form
from data_preprocessing import get_recent_form, calculate_goal_diff
import pandas as pd

def extract_features(df, home_team, away_team, match_date, le_team):
    home_team = TEAM_NAME_MAPPING.get(home_team.upper(), home_team.upper())
    away_team = TEAM_NAME_MAPPING.get(away_team.upper(), away_team.upper())

    past_home = get_past_matches(df, home_team, match_date)
    home_stats = calculate_team_stats(past_home, home_team)

    past_away = get_past_matches(df, away_team, match_date)
    away_stats = calculate_team_stats(past_away, away_team)

    h2h = head_to_head_stats(df, home_team, away_team, match_date)
    
    home_recent_form = get_recent_form(df, home_team, match_date)
    away_recent_form = get_recent_form(df, away_team, match_date)

    home_goal_diff = calculate_goal_diff(df, home_team, match_date)
    away_goal_diff = calculate_goal_diff(df, away_team, match_date)

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
        'home_recent_form': home_recent_form,
        'away_recent_form': away_recent_form,
        'home_goal_diff': home_goal_diff,
        'away_goal_diff': away_goal_diff,
    }])