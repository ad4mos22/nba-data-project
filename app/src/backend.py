import csv
import pandas as pd
import numpy as np

class Player:
    def __init__(self,player_origin_name , bbref_player_id, player_team, player_points, player_rebounds, player_assists):
        self.player_origin_name = player_origin_name
        self.bbref_player_id = bbref_player_id
        self.player_team = player_team
        self.player_points = player_points
        self.player_rebounds = player_rebounds
        self.player_assists = player_assists

class Game:
    def __init__(self, game_id, game_date, home_team, away_team, home_team_points, away_team_points):
        self.game_id = game_id
        self.game_date = game_date
        self.home_team = home_team
        self.away_team = away_team
        self.home_team_points = home_team_points
        self.away_team_points = away_team_points