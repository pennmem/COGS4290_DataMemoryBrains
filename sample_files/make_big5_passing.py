"""Generate `big5_passing_2022_2023.csv`, the player-passing dataset used by
Module 00, Problem 2.

The assignment originally scraped this table live from
https://fbref.com/en/comps/Big5/2022-2023/passing/players/... but fbref now returns
HTTP 403 to scripted requests, so the notebook could no longer be run. This script
produces a deterministic, fbref-shaped stand-in with the same column names and
plausible distributions.

If you obtain a real copy of the fbref table, save it over the CSV with the same
column names and the notebook will work unchanged.

    python sample_files/make_big5_passing.py
"""

import numpy as np
import pandas as pd

SEED = 20230601
N_PER_LEAGUE = 520

LEAGUES = ["eng Premier League", "es La Liga", "fr Ligue 1",
           "de Bundesliga", "it Serie A"]

SQUADS = {
    "eng Premier League": ["Arsenal", "Aston Villa", "Brentford", "Brighton", "Chelsea",
                           "Crystal Palace", "Everton", "Fulham", "Leeds United", "Leicester City",
                           "Liverpool", "Manchester City", "Manchester Utd", "Newcastle Utd",
                           "Nott'ham Forest", "Southampton", "Tottenham", "West Ham", "Wolves", "Bournemouth"],
    "es La Liga": ["Almería", "Athletic Club", "Atlético Madrid", "Barcelona", "Cádiz",
                   "Celta Vigo", "Elche", "Espanyol", "Getafe", "Girona", "Mallorca",
                   "Osasuna", "Rayo Vallecano", "Real Betis", "Real Madrid", "Real Sociedad",
                   "Sevilla", "Valencia", "Valladolid", "Villarreal"],
    "fr Ligue 1": ["Ajaccio", "Angers", "Auxerre", "Brest", "Clermont Foot", "Lens", "Lille",
                   "Lorient", "Lyon", "Marseille", "Monaco", "Montpellier", "Nantes", "Nice",
                   "Paris S-G", "Reims", "Rennes", "Strasbourg", "Toulouse", "Troyes"],
    "de Bundesliga": ["Augsburg", "Bayern Munich", "Bochum", "Dortmund", "Eint Frankfurt",
                      "Freiburg", "Hertha BSC", "Hoffenheim", "Köln", "Leverkusen", "Mainz 05",
                      "M'Gladbach", "RB Leipzig", "Schalke 04", "Stuttgart", "Union Berlin",
                      "Werder Bremen", "Wolfsburg"],
    "it Serie A": ["Atalanta", "Bologna", "Cremonese", "Empoli", "Fiorentina", "Hellas Verona",
                   "Inter", "Juventus", "Lazio", "Lecce", "Milan", "Monza", "Napoli", "Roma",
                   "Salernitana", "Sampdoria", "Sassuolo", "Spezia", "Torino", "Udinese"],
}

NATIONS = ["eng ENG", "es ESP", "fr FRA", "de GER", "it ITA", "br BRA", "ar ARG", "pt POR",
           "nl NED", "be BEL", "hr CRO", "rs SRB", "sn SEN", "ma MAR", "jp JPN", "us USA",
           "dk DEN", "no NOR", "se SWE", "pl POL", "ch SUI", "at AUT", "uy URU", "co COL"]

FIRST = ["Luca", "Marco", "Thomas", "Jordan", "Kevin", "Diego", "Andrés", "Mateo", "Youssef",
         "Ibrahim", "Nicolas", "Lucas", "Felix", "Jonas", "Erik", "Adam", "Noah", "Leo",
         "Rafael", "Bruno", "Pedro", "João", "Nuno", "Sergio", "Pablo", "Javier", "Iker",
         "Antoine", "Julien", "Maxime", "Théo", "Hugo", "Emil", "Viktor", "Anton", "Milan",
         "Stefan", "Daniel", "David", "Samuel", "Elias", "Omar", "Karim", "Yusuf", "Takumi",
         "Sota", "Ryan", "Connor", "Liam", "Owen", "Mason", "Callum", "Finn", "Oscar"]

LAST = ["Rossi", "Bianchi", "Müller", "Schmidt", "Fischer", "Bakker", "Jansen", "Silva",
        "Santos", "Pereira", "Costa", "Gómez", "Fernández", "Martínez", "López", "Sánchez",
        "Moreau", "Dubois", "Laurent", "Girard", "Petit", "Novak", "Horvat", "Kovač",
        "Nowak", "Kowalski", "Andersson", "Larsen", "Nielsen", "Hansen", "Berg", "Lindqvist",
        "Diallo", "Traoré", "Cissé", "Ndiaye", "Benali", "El Amrani", "Tanaka", "Yamamoto",
        "Walsh", "Doyle", "Byrne", "Murphy", "Clarke", "Hughes", "Wright", "Barnes",
        "Ferrari", "Conti", "Greco", "Rizzo", "Weber", "Wagner", "Becker", "Hoffmann"]

POSITIONS = ["MF", "MF,FW", "DF,MF", "MF,DF", "FW,MF", "DF", "FW", "GK", "DF,FW", "FW,DF"]
POS_WEIGHTS = [0.22, 0.07, 0.06, 0.04, 0.04, 0.28, 0.17, 0.09, 0.02, 0.01]


def make_dataframe(seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    rank = 1
    for league in LEAGUES:
        squads = SQUADS[league]
        for _ in range(N_PER_LEAGUE):
            pos = rng.choice(POSITIONS, p=POS_WEIGHTS)
            is_mf = "MF" in pos
            is_gk = pos == "GK"

            age = int(np.clip(rng.normal(26, 4.2), 16, 39))
            born = 2023 - age

            # playing time: heavily right-skewed, most players well under a full season
            nineties = float(np.round(np.clip(rng.gamma(2.0, 5.0), 0.0, 38.0), 1))

            # passing volume scales with minutes and position
            per90_att = rng.normal(70 if is_gk else (55 if is_mf else 42), 12)
            per90_att = max(per90_att, 8.0)
            total_att = int(max(np.round(per90_att * nineties), 0))

            # completion % — midfielders and keepers differ from the rest
            base_cmp = 85.0 if is_mf else (72.0 if is_gk else 79.0)
            cmp_pct = float(np.clip(rng.normal(base_cmp, 5.5), 40.0, 98.0))
            total_cmp = int(np.round(total_att * cmp_pct / 100.0))
            cmp_pct = float(np.round(100.0 * total_cmp / total_att, 1)) if total_att else np.nan

            tot_dist = int(np.round(total_cmp * rng.normal(19.5, 2.4))) if total_cmp else 0
            prg_dist = int(np.round(tot_dist * np.clip(rng.normal(0.34, 0.07), 0.05, 0.75)))

            short_att = int(np.round(total_att * np.clip(rng.normal(0.46, 0.06), 0.1, 0.85)))
            medium_att = int(np.round(total_att * np.clip(rng.normal(0.38, 0.06), 0.1, 0.8)))
            long_att = max(total_att - short_att - medium_att, 0)
            short_cmp = int(np.round(short_att * np.clip(rng.normal(0.92, 0.04), 0.3, 1.0)))
            medium_cmp = int(np.round(medium_att * np.clip(rng.normal(0.87, 0.05), 0.3, 1.0)))
            long_cmp = int(np.round(long_att * np.clip(rng.normal(0.58, 0.10), 0.1, 1.0)))

            kp_rate = rng.gamma(2.0, 0.55 if is_mf else 0.30)      # key passes per 90
            key_passes = int(np.round(kp_rate * nineties))
            assists = int(rng.poisson(max(key_passes * 0.09, 0.01)))
            xag = float(np.round(max(rng.normal(assists, 1.1), 0.0), 1))
            xa = float(np.round(max(xag + rng.normal(0, 0.4), 0.0), 1))

            rows.append({
                "Rk": rank,
                "Player": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "Nation": rng.choice(NATIONS),
                "Pos": pos,
                "Squad": rng.choice(squads),
                "Comp": league,
                "Age": age,
                "Born": born,
                "90s": nineties,
                "Total_Cmp": total_cmp,
                "Total_Att": total_att,
                "Total_Cmp%": cmp_pct,
                "Total_TotDist": tot_dist,
                "Total_PrgDist": prg_dist,
                "Short_Cmp": short_cmp,
                "Short_Att": short_att,
                "Short_Cmp%": round(100.0 * short_cmp / short_att, 1) if short_att else np.nan,
                "Medium_Cmp": medium_cmp,
                "Medium_Att": medium_att,
                "Medium_Cmp%": round(100.0 * medium_cmp / medium_att, 1) if medium_att else np.nan,
                "Long_Cmp": long_cmp,
                "Long_Att": long_att,
                "Long_Cmp%": round(100.0 * long_cmp / long_att, 1) if long_att else np.nan,
                "Ast": assists,
                "xAG": xag,
                "xA": xa,
                "A-xAG": round(assists - xag, 1),
                "KP": key_passes,
                "1/3": int(np.round(total_cmp * np.clip(rng.normal(0.10, 0.03), 0, 0.4))),
                "PPA": int(np.round(key_passes * np.clip(rng.normal(1.4, 0.4), 0, 4))),
                "CrsPA": int(np.round(key_passes * np.clip(rng.normal(0.35, 0.2), 0, 2))),
                "PrgP": int(np.round(total_cmp * np.clip(rng.normal(0.09, 0.03), 0, 0.4))),
                "Matches": "Matches",
            })
            rank += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_dataframe()
    out = "sample_files/big5_passing_2022_2023.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {df.shape[0]} rows x {df.shape[1]} cols")
