
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
# Load data
df_circuit = pd.read_csv('circuit1.csv')
df_constructor_result = pd.read_csv('constructor_results.csv')
df_constructor_standing = pd.read_csv('constructor_standings.csv')
df_constructor = pd.read_csv('constructors.csv')
df_driver_standing = pd.read_csv('driver_standings.csv')
df_drivers = pd.read_csv('drivers.csv')
df_drivers["Driver_name"] = df_drivers["forename"] + ' ' + df_drivers['surname']
df_lap_times = pd.read_csv('lap_times.csv')
df_pit_stops = pd.read_csv('pit_stops.csv')
df_qualifying = pd.read_csv('qualifying.csv')
df_races = pd.read_csv('races.csv')
df_results = pd.read_csv('results.csv')
df_seasons = pd.read_csv('seasons.csv')
df_sprint_results = pd.read_csv('sprint_results.csv')
df_status = pd.read_csv('status.csv')
df_picture = pd.read_excel('picture.xlsx')
df_layout = pd.read_excel('Layout.xlsx')
df_car = pd.read_excel("cars.xlsx")
df_final = pd.read_csv("df_final_2.csv")

st.markdown("""<style>
   button[data-baseweb="tab"] {
   font-size: 24px;
   margin: 0;
   width: 100%;
   }
</style>""", unsafe_allow_html=True)

st.markdown("""
<style>.element-container:has(#button-after) + div button {
width: 200px;
text-align: center;
padding: 15px;
margin: auto;
display: block;
}</style>""", unsafe_allow_html=True)

def create_map():

    temp = df_results.merge(df_races, on='raceId')
    temp = temp[['circuitId', 'driverId', "constructorId", "positionOrder", "year"]]
    temp = df_circuit.merge(temp, on='circuitId')
    temp1 = df_drivers[["Driver_name", "driverId"]]
    temp = temp.merge(temp1, on='driverId')
    temp_cons = df_constructor[['constructorId', 'name']]
    temp_cons = temp_cons.rename(columns={'name': 'Constructor_name'})
    temp = temp.merge(temp_cons, on='constructorId')

    # Create a Map instance
    m = folium.Map(location=[20, 0], zoom_start=2)

    # Suppose you have a DataFrame df_pilotes with columns 'circuit_id', 'pilote_name', 'position', 'year', 'name', 'Length', and 'Turns'
    # Sort the DataFrame by 'position' in descending order to get the last 5 winners for each circuit
    df_pilotes_sorted = temp[temp['positionOrder'] == 1].sort_values(by=['circuitId','year'], ascending=False)

    # Get unique years in the dataset
    unique_years = sorted(df_pilotes_sorted['year'].unique(),reverse=True)

    # Group the sorted DataFrame by 'circuit_id' and get the last 5 winners for each circuit
    last_5_winners = df_pilotes_sorted.groupby('circuitId').head(5)

    # Function to format popup text with HTML table
    def get_popup_text(row):
        popup_text = f"<h3 style='margin-bottom: 10px;'>{row['location']}, {row['country']}</h3>"

        # Add circuit information
        popup_text += f"<b>Length:</b> {row['Length']} km<br>"
        popup_text += f"<b>Turns:</b> {row['Turns']}<br>"

        popup_text += f"<h3 style='margin-bottom: 10px;'></h3>"

        popup_text += "<b>Last 5 winners:</b><br>"
        popup_text += "<table style='border-collapse: collapse; width: 100%; margin-top: 10px;'>"
        popup_text += "<tr style='background-color: #f2f2f2;'>"
        popup_text += "<th style='border: 1px solid #dddddd; text-align: left; padding: 8px;'>Driver</th>"
        popup_text += "<th style='border: 1px solid #dddddd; text-align: left; padding: 8px;'>Constructor</th>"
        popup_text += "<th style='border: 1px solid #dddddd; text-align: left; padding: 8px;'>Year</th>"
        popup_text += "</tr>"

        # Get the last 5 winners for the current circuit
        last_5_winners_circuit = last_5_winners[last_5_winners['circuitId'] == row['circuitId']]
        for _, winner_row in last_5_winners_circuit.iterrows():
            popup_text += "<tr>"
            popup_text += f"<td style='border: 1px solid #dddddd; padding: 8px;'>{winner_row['Driver_name']}</td>"
            popup_text += f"<td style='border: 1px solid #dddddd; padding: 8px;'>{winner_row['Constructor_name']}</td>"
            popup_text += f"<td style='border: 1px solid #dddddd; padding: 8px;'>{winner_row['year']}</td>"
            popup_text += "</tr>"

        popup_text += "</table>"

        return popup_text

    # Function to filter circuits for a specific year
    def filter_circuits(year):
        filtered_circuits = df_pilotes_sorted[df_pilotes_sorted['year'] == year]

        # Clear existing markers on the map
        m = folium.Map(location=[20, 0], zoom_start=2)

        # Iterate over filtered circuits DataFrame to add markers with popups
        for idx, row in filtered_circuits.iterrows():
            popup_text = get_popup_text(row)
            folium.Marker(
                location=[row['lat'], row['lng']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=row['name'],  # Display circuit name as tooltip
                icon=folium.Icon(color='red', icon='info-sign')  # Set icon color and style
            ).add_to(m)

        # Add custom CSS to style the popup background color
        css = """
        <style>
        .leaflet-popup-content-wrapper {
            background-color: #333;
            color: #fff;
        }
        </style>
        """
        folium.Popup(css).add_to(m)

        st.markdown(f"<div style='text-align: center;'><h2>Circuit Map for {year}</h2></div>", unsafe_allow_html=True)
        folium_static(m)

    # Get current year from session state or default to the last year in the dataset
    current_year = st.session_state.get("current_year", unique_years[-1])

    # Update current year in session state
    st.session_state.current_year = st.selectbox("Select Year", options=unique_years, index=unique_years.index(current_year), key="year_selector")

    # Call the function when the year is selected
    filter_circuits(st.session_state.current_year)

def get_constructor_standings(year, round_num):
    # 1. Filtrer les données pour l'année spécifiée
    races_current_year = df_races[(df_races['year'] == year) & (df_races['round'] <= round_num)]

    # 2. Identifier le dernier round jusqu'au round spécifié
    last_round = races_current_year['round'].max()

    # 3. Filtrer les données pour ce round spécifié
    round_data = races_current_year[races_current_year['round'] == round_num]

    # 4. Fusionner les données des classements des constructeurs pour ce round spécifié
    standing = round_data.merge(df_constructor_standing, on='raceId')

    constructor = df_constructor[['constructorId','name','nationality']].rename(columns={'name':'F1 Team'})

    standing = standing.merge(constructor,on='constructorId')

    # 5. Trier les données par points des constructeurs dans l'ordre décroissant
    standing = standing.sort_values(by='points', ascending=False)
    # Réinitialiser l'index pour retirer les index par défaut
    standing.reset_index(drop=True, inplace=True)
    return standing[['position', 'F1 Team', 'points', 'wins']]

def get_driver_standings(year, round_num):
    # 1. Filtrer les données pour l'année spécifiée
    races_current_year = df_races[(df_races['year'] == year) & (df_races['round'] <= round_num)]

    # 2. Identifier le dernier round jusqu'au round spécifié
    last_round = races_current_year['round'].max()

    # 3. Filtrer les données pour ce round spécifié
    round_data = races_current_year[races_current_year['round'] == round_num]

    # 4. Fusionner les données des classements des constructeurs pour ce round spécifié
    standing = round_data.merge(df_driver_standing, on='raceId')

    drivers = df_drivers[['driverId','Driver_name','nationality']]

    standing = standing.merge(drivers,on='driverId')


    # 5. Trier les données par points des constructeurs dans l'ordre décroissant
    standing = standing.sort_values(by='points', ascending=False).sort_values(by='position', ascending=True)

    return standing[['position', 'Driver_name','points', 'wins']]

def plot_constructor_standings(year):
    # Filtrer les données pour l'année spécifiée
    races_current_year = df_races[df_races['year'] == year]

    # Fusionner les données des classements des constructeurs pour toute l'année
    standing = races_current_year.merge(df_constructor_standing, on='raceId')

    constructor = df_constructor[['constructorId','name','nationality']].rename(columns={'name':'F1 Team'})
    standing = standing.merge(constructor,on='constructorId')

    # Trier les données par round
    standing = standing.sort_values(by=['round', 'position'], ascending=True)

    # Définir les couleurs pour les équipes spécifiées
    color_map = {
        'Ferrari': 'red',
        'Mercedes': 'lightblue',
        'Red Bull': 'darkblue',
        'McLaren': 'orange'
    }

    # Créer une figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Traiter les données pour la visualisation
    for team, data in standing.groupby('F1 Team'):
        color = color_map.get(team, 'gray')  # Utiliser la couleur spécifiée ou grise par défaut
        ax.plot(data['round'], data['points'], label=team, color=color)

    # Ajouter des étiquettes, une légende, un titre, etc.
    ax.set_xlabel('Round')
    ax.set_ylabel('Points')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)
    st.pyplot(fig)
    
def plot_driver_standings(year):
    # Filtrer les données pour l'année spécifiée
    races_current_year = df_races[df_races['year'] == year]

    # Fusionner les données des classements des constructeurs pour toute l'année
    standing = races_current_year.merge(df_driver_standing, on='raceId')

    drivers = df_drivers[['driverId','Driver_name','nationality']]

    standing = standing.merge(drivers,on='driverId')

    # Trier les données par round
    standing = standing.sort_values(by=['round', 'position'], ascending=True)

    # Créer une figure et un axe
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("colorblind", len(standing['Driver_name'].unique()))
    
    # Parcourir chaque constructeur pour tracer sa progression de points
    for i, (team, data) in enumerate(standing.groupby('Driver_name')):
        ax.plot(data['round'], data['points'], label=team, color=palette[i])

    # Ajouter des étiquettes, une légende, un titre, etc.
    ax.set_xlabel('Round')
    ax.set_ylabel('Points')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3)
    st.pyplot(fig)

def plot_completion_rate(driver_name, year_start, year_end):
    # Filtrer les résultats pour le pilote spécifié
    driver_results = df_results.merge(df_drivers, on='driverId').merge(df_races, on= 'raceId')
    driver_results = driver_results[(driver_results['Driver_name'] == driver_name) &
                                ((driver_results["year"] >= year_start) & (driver_results['year'] <= year_end))]

    # Calculer le pourcentage de courses terminées
    total_races = len(driver_results)
    completed_races = len(driver_results[driver_results['positionText'] != 'R'])
    completion_rate = (completed_races / total_races) * 100

    # Dessiner la jauge
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.pie([completion_rate, 100 - completion_rate], colors=['red', 'lightgrey'], startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.text(0, 0, f'{completion_rate:.2f}%', ha='center', va='center', fontsize=20, color='black')
    ax.text(0, -0.3, 'Races finished', ha='center', va='center', fontsize=12, color='black')
    ax.axis('off')
    st.pyplot()

def plot_podium_percentage(driver_name, year_start, year_end):
    # Filtrer les résultats pour le pilote spécifié
    driver_results = df_results.merge(df_drivers, on='driverId').merge(df_races, on= 'raceId')
    driver_results = driver_results[(driver_results['Driver_name'] == driver_name) &
                                ((driver_results["year"] >= year_start) & (driver_results['year'] <= year_end))]

    # Calculer le pourcentage de podiums
    total_races = len(driver_results)
    podium_races = len(driver_results[(driver_results['positionText'] == '1') |
                                      (driver_results['positionText'] == '2') |
                                      (driver_results['positionText'] == '3')])
    podium_percentage = (podium_races / total_races) * 100

    # Dessiner la jauge
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.pie([podium_percentage, 100 - podium_percentage], colors=['red', 'lightgrey'], startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.text(0, 0, f'{podium_percentage:.2f}%', ha='center', va='center', fontsize=20, color='black')
    ax.text(0, -0.3, 'Podiums', ha='center', va='center', fontsize=12, color='black')
    ax.axis('off')
    st.pyplot()

def plot_driver_avg_points(selected_year, selected_driver):
    # Filtrer les résultats pour l'année sélectionnée
    driver_results = df_results.merge(df_drivers, on='driverId').merge(df_races, on= 'raceId')
    year_results = driver_results[driver_results['year'] == selected_year]
    
    # Calculer le nombre de points moyen pour chaque pilote
    avg_points = year_results.groupby('Driver_name')['points'].mean().reset_index()
    
    # Créer un barplot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_points['Driver_name'], avg_points['points'], color='lightgrey')
    
    # Mettre en rouge la barre du pilote sélectionné
    for bar in bars:
        if bar.get_height() == avg_points.loc[avg_points['Driver_name'] == selected_driver, 'points'].values:
            bar.set_color('red')
    
    # Ajouter des titres et libellés
    plt.xlabel('Driver')
    plt.ylabel('Avg Points')
    plt.xticks(rotation=45, ha='right')
    
    # Afficher le graphique
    plt.tight_layout()
    st.pyplot()
    
def plot_points_by_year(driver_name):
    race_year = df_races[['raceId', 'year']]
    # Agrégation des points par année pour le pilote spécifié
    point_history = df_results.merge(df_drivers, on="driverId")
    point_history = point_history.merge(race_year, on='raceId')
    point_history = point_history[(point_history['Driver_name'] == driver_name)]
    points_by_year = point_history.groupby('year')['points'].sum().reset_index()

    # Création du graphique à barres
    plt.figure(figsize=(10, 6))
    plt.bar(points_by_year['year'], points_by_year['points'], color='red')
    plt.xlabel('Year')
    plt.ylabel('Points')
    plt.xticks(points_by_year['year'])  # Assurez-vous que toutes les années sont affichées sur l'axe des x
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot()
    
def plot_lap_times(year, round_num, selected_driver):
    t = df_lap_times.merge(df_races[['raceId','round','name', 'year']], on = 'raceId').merge(df_drivers[["Driver_name","driverId"]], on = 'driverId')
    t['milliseconds'] = t['milliseconds']/1000
    # Filtrer les données pour l'année et le round spécifiés
    filtered_df = t[(t['year'] == year) & (t['round'] == round_num)]

    # Créer le box plot
    plt.figure(figsize=(12, 8))
    # Créer le boxplot sans afficher les outliers
    palette = {selected_driver: 'red'}
    for driver in filtered_df['Driver_name'].unique():
        if driver != selected_driver:
            palette[driver] = 'lightgrey'
            
    sns.boxplot(x='milliseconds', y='Driver_name', data=filtered_df, showfliers=False, palette=palette)


    plt.title(f'Lap Times Distribution - Year {year}, Round {round_num}')
    plt.xlabel('Lap Time (seconds)')
    plt.ylabel('Driver')
    plt.tight_layout()
    st.pyplot()
    
def engine_failures_vs_altitude():
    df = df_races[['raceId','year','round','circuitId','name']]
    df1 = df_circuit[['circuitId','name','alt','Length','Turns']]
    df2 = df_results[['raceId','driverId','constructorId','statusId','position']]
    
    df3 = df.merge(df1, on='circuitId').rename(columns={'name_x': 'Grand_prix', 'name_y': 'Circuit_name'})
    df3 = df3.merge(df2, on='raceId')
    df3 = df3.merge(df_status, on='statusId')
    
    alt_effect = df3[df3['status'].isin(['Transmission','Engine','Overheating','Engine fire','Power loss'])]
    alt_effect = alt_effect[alt_effect['year'] >= 2011]
    alt_effect = alt_effect.groupby(['Circuit_name','alt'])['status'].count().sort_values(ascending=False).reset_index().head(10)
    alt_effect = alt_effect.rename(columns={'status':'Engine related failures'})
    
    fig = px.scatter(alt_effect,
                     x="alt",
                     y="Engine related failures",
                     size="alt",
                     color="Circuit_name",
                     log_x=True,
                     size_max=50,
                     labels={"alt": "Altitude (m)", "Engine related failures": "Failures"},
                     title="Engine Failures vs Altitude by Circuit"
                    )

    fig.update_traces(marker=dict(line=dict(color='#000000', width=1.49)))
    
    fig.update_layout(
        title=dict(text="Engine Failures vs Altitude by Circuit", font=dict(size=18, family="Arial"), x=0.25), # Centrage et taille du titre
        font=dict(size=14, family="Arial"),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            title_text="Altitude (m)",
            title_font=dict(size=18, family="Arial"),
            type="linear",
            showline=True,    # Afficher la ligne de l'axe x
            linewidth=2,      # Épaisseur de la ligne de l'axe x
            linecolor='black' # Couleur de la ligne de l'axe x
        ),
        yaxis=dict(
            showgrid=False,
            title_text="Failures",
            title_font=dict(size=18, family="Arial"),
            showline=True,    # Afficher la ligne de l'axe y
            linewidth=2,      # Épaisseur de la ligne de l'axe y
            linecolor='black' # Couleur de la ligne de l'axe y
        ),
        plot_bgcolor='rgba(255, 255, 255, 0.9)',
        width=650
    )
    
    st.plotly_chart(fig)

def calculate_historic_team_wins(year_start,year_end,selected_team):
    """
    Calcule le nombre de victoires par équipe à partir de l'année spécifiée.

    Args:
    - year_start (int): Année à partir de laquelle calculer les victoires.

    Returns:
    - DataFrame: DataFrame contenant le nombre de victoires par équipe.
    """

    # Obtenir les données sur l'année de chaque course
    race_year = df_races[['raceId', 'year']]

    # Fusionner les données sur les résultats des courses avec les données sur les constructeurs et les années de course
    point_per_team = df_results.merge(df_constructor, on="constructorId")
    point_per_team = point_per_team.merge(race_year, on='raceId')

    # Filtrer les données pour inclure uniquement les courses à partir de l'année spécifiée
    point_per_team = point_per_team[((point_per_team['year'] >= year_start) & (point_per_team['year'] <= year_end)) & (point_per_team['name'] == selected_team)]

    # Filtrer les données pour inclure uniquement les victoires
    point_per_team = point_per_team[(point_per_team['position'] == '1')]
    
    win_count = len(point_per_team)
    
    return win_count


def calculate_historic_team_point(year_start,year_end,selected_team):
    """
    Calcule le nombre de victoires par équipe à partir de l'année spécifiée.

    Args:
    - year_start (int): Année à partir de laquelle calculer les victoires.

    Returns:
    - DataFrame: DataFrame contenant le nombre de victoires par équipe.
    """

    # Obtenir les données sur l'année de chaque course
    race_year = df_races[['raceId', 'year']]

    # Fusionner les données sur les résultats des courses avec les données sur les constructeurs et les années de course
    point_per_team = df_results.merge(df_constructor, on="constructorId")
    point_per_team = point_per_team.merge(race_year, on='raceId')

    # Filtrer les données pour inclure uniquement les courses à partir de l'année spécifiée
    point_per_team = point_per_team[((point_per_team['year'] >= year_start) & (point_per_team['year'] <= year_end)) & (point_per_team['name'] == selected_team)]

    # Filtrer les données pour inclure uniquement les victoires
    point_per_team = point_per_team["points"].sum()
    
    return point_per_team

def plot_completion_team_rate(name, year_start, year_end):
    # Filtrer les résultats pour le pilote spécifié
    driver_results = df_results.merge(df_constructor, on='constructorId').merge(df_races, on= 'raceId')
    driver_results = driver_results[(driver_results['name_x'] == name) &
                                ((driver_results["year"] >= year_start) & (driver_results['year'] <= year_end))]

    # Calculer le pourcentage de courses terminées
    total_races = len(driver_results)
    completed_races = len(driver_results[driver_results['positionText'] != 'R'])
    completion_rate = (completed_races / total_races) * 100

    # Dessiner la jauge
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.pie([completion_rate, 100 - completion_rate], colors=['red', 'lightgrey'], startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.text(0, 0, f'{completion_rate:.2f}%', ha='center', va='center', fontsize=20, color='black')
    ax.text(0, -0.3, 'Races finished', ha='center', va='center', fontsize=12, color='black')
    ax.axis('off')
    st.pyplot()

def plot_podium_team_percentage(name, year_start, year_end):
    # Filtrer les résultats pour le pilote spécifié
    race_year = df_races[['raceId', 'year']]
    driver_results = df_results.merge(df_constructor, on='constructorId')
    driver_results = driver_results.merge(race_year, on = 'raceId')
    driver_results = driver_results[driver_results['name'] == name]
    driver_results = driver_results[(driver_results["year"] >= year_start) & (driver_results['year'] <= year_end)]

    # Calculer le pourcentage de podiums
    total_races = len(driver_results)
    podium_races = len(driver_results[((driver_results['positionText'] == '1') |
                                      (driver_results['positionText'] == '2') |
                                      (driver_results['positionText'] == '3'))])

    podium_percentage = (podium_races / total_races) * 100

    # Dessiner la jauge
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.pie([podium_percentage, 100 - podium_percentage], colors=['red', 'lightgrey'], startangle=90,
           wedgeprops=dict(width=0.3, edgecolor='w'))
    ax.text(0, 0, f'{podium_percentage:.2f}%', ha='center', va='center', fontsize=20, color='black')
    ax.text(0, -0.3, 'Podiums', ha='center', va='center', fontsize=12, color='black')
    ax.axis('off')
    st.pyplot()

def plot_team_positions_by_year(year, selected_team):
    # Sélectionner les données pour l'année spécifiée
    race_year = df_races[['raceId', 'year']]
    position_per_team = df_results.merge(df_constructor, on="constructorId")
    position_per_team = position_per_team.merge(race_year, on='raceId')
    position_per_team = position_per_team[(position_per_team['year'] == year)]

    # Tracer le boxplot avec des couleurs différentes pour chaque équipe
    fig, ax = plt.subplots(figsize=(10, 8))

    # Définir la palette de couleurs
    palette = {team: "red" if team == selected_team else "grey" for team in position_per_team['name'].unique()}

    sns.boxplot(x="positionOrder", y="name", data=position_per_team,
                ax=ax, palette=palette, showfliers = False)
    ax.set_xlabel("Finish position")
    ax.set_ylabel("Team")

    # Définir les emplacements des marqueurs sur l'axe des abscisses comme des entiers
    plt.xticks(range(1, position_per_team["positionOrder"].max() + 1, 2))

    st.pyplot()

def boxplot_by_team_for_year(year, selected_team):
    t = df_races.merge(df_pit_stops, on='raceId')
    t = t.merge(df_drivers, on='driverId')
    t = t[['raceId', 'year', 'round', 'driverId', 'milliseconds']]
    t = t.merge(df_results[['raceId', 'driverId', 'constructorId', 'positionOrder']], on=['raceId', 'driverId'])
    t = t.merge(df_constructor[['constructorId', 'name']], on='constructorId')

    t['milliseconds'] = t['milliseconds'] / 1000
    t = t[t['milliseconds'] <= 60]

    # Filtrer les données pour l'année spécifiée
    df_year = t[t['year'] == year]

    # Créer une liste unique d'équipes
    teams = df_year['name'].unique()

    # Créer un dictionnaire pour stocker les données par équipe
    team_data = {team: [] for team in teams}

    # Ajouter les données de temps de pit stop pour chaque équipe dans le dictionnaire
    for team in teams:
        team_data[team] = df_year[df_year['name'] == team]['milliseconds']

    # Définir une liste de couleurs pour chaque équipe
    colors = ['grey' if team != selected_team else 'red' for team in teams]

    # Créer un graphique à boîte à moustaches pour chaque équipe
    plt.figure(figsize=(10, 10))
    bp = plt.boxplot(team_data.values(), labels=team_data.keys(), vert=False, patch_artist=True, showfliers=False)

    # Ajouter des couleurs aux boîtes à moustaches
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)

    plt.ylabel('Team')
    plt.xlabel('Pit stop (seconds)')
    plt.title(f'Mechanic crew performance during pit-stops in {year}')
    st.pyplot()

    
# Function for the homepage
def home():
    # Centered title for the homepage
    st.markdown("<div style='text-align: center;'><h1>Welcome to our Formula 1 Application</h1></div>", unsafe_allow_html=True)
    st.write("Select an option from the sidebar to get started.")
    
# Function for the current season tab


def current_season():
    st.markdown("<div style='text-align: center;'><h1>Formula 1's Season</h1></div>", unsafe_allow_html=True)

    # Create a list of years available in the dataset
    years_available = sorted(df_races['year'].unique(), reverse=True)
    
    # Display a SelectBox to choose the year
 
    selected_year, selected_round = st.columns(2)  # Diviser l'écran en 2 colonnes
    with selected_year:
        selected_year = st.selectbox("Select Year", years_available)
    with selected_round:
        r = sorted(df_races[df_races['year'] == selected_year]['round'].unique())
        selected_round = st.selectbox("Select Round", r)
    # Call get_constructor_standings() with the selected year
    constructor_standings_current_season = get_constructor_standings(selected_year,selected_round)
    driver_standings_current_season = get_driver_standings(selected_year,selected_round)

    # Display the constructor standings for the selected year
    
    def color_rows(val):
        if val.name < 3:  
            if val.name == 0:  
                return ['background-color: gold'] * len(val)
            elif val.name == 1:  
                return ['background-color: silver'] * len(val)
            elif val.name == 2:  
                return ['background-color: #cd7f32'] * len(val)  
        return [''] * len(val)
    
    def color_rows_driver(val):
        if val['position'] == 1:
            return ['background-color: gold'] * len(val)
        elif val['position'] == 2:
            return ['background-color: silver'] * len(val)
        elif val['position'] == 3:
            return ['background-color: #cd7f32'] * len(val)
        return [''] * len(val)
    
    # Formatage des données pour la colonne 'Points'
    constructor_standings_current_season['points'] = constructor_standings_current_season['points'].apply(lambda x: '{:.1f}'.format(x))
    driver_standings_current_season['points'] = driver_standings_current_season['points'].apply(lambda x: '{:.1f}'.format(x))
    

    # Fusionner les données des classements des constructeurs pour toute l'année
    
    nb_driver = df_races.merge(df_driver_standing, on='raceId')
    country = df_races.merge(df_circuit, on='circuitId')
    standing = nb_driver.merge(df_drivers,on='driverId')
    nb_driver = nb_driver[nb_driver['year'] == selected_year]['driverId'].unique()
    nb_nationality = standing[(standing['year'] == selected_year)]['nationality'].unique()
    race_name = df_races[(df_races['year'] == selected_year) & (df_races['round'] == selected_round)]['name'].iloc[0]
    country = country[(country['year'] == selected_year) & (country['round'] == selected_round)]['country'].iloc[0]
    
    with st.expander(f"Brief recap of {selected_year}'s season"):
        c1, c2, c3= st.columns(3)
        c1.metric("Number of race :racing_car:", len(r))
        c2.metric("Number of driver", len(nb_driver))
        c3.metric("Number of nationality", len(nb_nationality))
    
    with st.expander(f"Round {selected_round} details"):
        c6,c4, c5 = st.columns(3)
        c6.metric("Country", country)
        c4.metric("Grand Prix :checkered_flag:", race_name)
        c5.metric("Weather :sunny::rain_cloud:", "A faire")
    
    
    col_1, col_2 = st.columns(2)
    
    with col_1:
    # Affichage du DataFrame avec le style personnalisé
        with st.expander(f"Standings for each race of {selected_year}'s season"):
          # Diviser l'écran en 2 colonnes
        
            st.write(f"<div style='text-align: left;'>Constructor Standings at the end of round {selected_round}:</div>", unsafe_allow_html=True)
            st.dataframe(constructor_standings_current_season.style.apply(color_rows, axis=1), 
                     hide_index=True)
    with col_2:
        with st.expander(f"Standings for each race of {selected_year}'s season"):

            st.write(f"<div style='text-align: left;'>Driver Standings at the end of round {selected_round}:</div>", unsafe_allow_html=True)
            st.dataframe(driver_standings_current_season.style.apply(color_rows_driver, axis=1), 
                     hide_index=True)
        
    
    
    col1, col2 = st.tabs(["Constructor Championship progression 📈 ", "Driver Championship progression 📈"])  # Diviser l'écran en 2 colonnes

    
    with col1:
        st.markdown(f"<h1 style='text-align: center;'>Constructor Championship progression - {selected_year}</h1>", unsafe_allow_html=True)
        plot_constructor_standings(selected_year)
    with col2:
        st.write(f"<h1 style='text-align: center;'>Driver Championship progression - {selected_year}</h1>", unsafe_allow_html=True)
        plot_driver_standings(selected_year)
    



st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 20px;
}

[data-testid="stMetricLabel"] {
    font-size: 20px;
    text-align: center;
}

.st-emotion-cache-1jmvea6 {
    margin: 0 auto; /* Centrage horizontal */
    text-align: center; /* Centrage horizontal pour le contenu texte */
}
</style>
""",
    unsafe_allow_html=True,
)
# Function for the drivers' stats tab
def driver_stats():
    st.markdown("<div style='text-align: center;'><h1>Driver's statistics</h1></div>", unsafe_allow_html=True)
    
    with st.expander("All time record"):
        c,c1, c2, c3= st.columns(4)
        c.metric("Most World Titles :trophy:", "Ham & Schu", "7 titles")
        c1.metric("Most wins :sports_medal:", "Lewis Hamilton ","103 wins")
        c2.metric("Most poles positions :first_place_medal:", "Lewis Hamilton " , "104 wins")
        c3.metric("Most laps led :racing_car:", "Lewis Hamilton", "5455 laps" )
    
    df_photo = df_drivers.merge(df_picture, on='Driver_name', how='left')
    df_photo = df_photo.sort_values(by="Photo url")
    df_photo['Photo url'] = df_photo['Photo url'].fillna('https://raw.githubusercontent.com/Mohbadi/AppF1/main/Picture/unknown.jpg')
    
    
    # Affichage de la liste des pilotes
    selected_pilote = st.selectbox("Select a driver :", df_photo['Driver_name'])
    
    # Affichage de la photo du pilote sélectionné
    pilote_row = df_photo[df_photo['Driver_name'] == selected_pilote]
    pilote_photo_url = pilote_row.iloc[0]['Photo url']
    col1, col2 = st.columns(2)
    with col1:
        st.image(pilote_photo_url, width=296)
        st.markdown(f'<div style="border: 1px solid black; padding: 5px; text-align: center">{selected_pilote} - {df_photo[df_photo["Driver_name"]==selected_pilote]["number"].iloc[0]}</div>', unsafe_allow_html=True)
    
    
    team = df_results.merge(df_drivers, on = "driverId").merge(df_constructor, on = 'constructorId').merge(df_races, on='raceId').sort_values(by='year', ascending=False)
    
    data = {
    "Date of birth": [df_photo[df_photo['Driver_name']==selected_pilote]['dob'].iloc[0]],
    "Current Team": [team[team['Driver_name']==selected_pilote]['name_x'].iloc[0]],
    "Nationality": [df_photo[df_photo['Driver_name']==selected_pilote]['nationality'].iloc[0]],
    "Podiums": [len(team[(team['Driver_name']==selected_pilote) & (team['positionOrder'] <= 3)])],
    "Points (Until last race)": [team[(team['Driver_name']==selected_pilote)]['points'].sum()],
    "Grands Prix entered": [len(team[(team['Driver_name']==selected_pilote)]['raceId'].unique())],
    "Wins": [len(team[(team['Driver_name']==selected_pilote) & (team['positionOrder'] == 1)])],
    "Highest race finish": [team[(team['Driver_name']==selected_pilote)]['positionOrder'].min()],
    "Highest grid position":[team[(team['Driver_name']==selected_pilote) & (team['grid'] != 0)]['grid'].min()]
    }
    

    df_team = pd.DataFrame(data)

    # Transposer le DataFrame pour mettre les équipes sur les lignes
    df_team = df_team.T
    
    # Réinitialiser les index
    df_team.reset_index(inplace=True)
    
    # Renommer les colonnes
    df_team.columns = ["Category", "  _______________________  "]
    with col2:
        st.dataframe(df_team, 
             hide_index=True)
    
    st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 20px;'>Points scored by {selected_pilote} during his career</h1></div>", unsafe_allow_html=True)
    plot_points_by_year(selected_pilote)
    
    with st.expander(f"Deeper statistics on {selected_pilote}"):
        
        intervale = sorted(team[team['Driver_name'] == selected_pilote]["year"].unique(), reverse= False)
        start_year, end_year = st.select_slider(
        "Select a range of year",
        options= intervale,
        value= [intervale[0],intervale[-1]])
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>Race finished rate between {start_year} and {end_year}</h1></div>", unsafe_allow_html=True)
            plot_completion_rate(selected_pilote, start_year, end_year)
        with c2:
            st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>Podium finishes rate between {start_year} and {end_year}</h1></div>", unsafe_allow_html=True)
            plot_podium_percentage(selected_pilote, start_year, end_year)
        
        st.markdown('---')
        selected_year = st.selectbox("Select a year :", sorted(team[team['Driver_name'] == selected_pilote]["year"].unique(), reverse= False))
        st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>Average points scored during the {selected_year}'s season by {selected_pilote} compared to the others</h1></div>", unsafe_allow_html=True)
        plot_driver_avg_points(selected_year, selected_pilote)
        
        st.markdown('---')
        col1,col2 = st.columns(2)
        with col1:
            year = st.selectbox("Select a year :", sorted(team[team['Driver_name'] == selected_pilote]["year"].unique(), reverse= False), key='year')
        with col2:
            round_num = st.selectbox("Select a round :", sorted(team[(team['Driver_name'] == selected_pilote) & (team['year'] == year)]["round"].unique(), reverse= False))
        gp = team[(team['year'] == year) & (team['round'] == round_num)]['name_y'].iloc[0]
        st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>{selected_pilote}'s performance over laps at the {gp} compared to the others</h1></div>", unsafe_allow_html=True)

        plot_lap_times(year, round_num, selected_pilote)

# Function for the constructors' stats tab
def constructor_stats():
    st.title("Constructor Statistics")
    st.write("Constructor statistics will be displayed here.")
    
    
    df_photo = df_constructor.merge(df_car, on='name', how='left')
    df_photo = df_photo.sort_values(by="Photo url")
    df_photo['Photo url'] = df_photo['Photo url'].fillna('https://raw.githubusercontent.com/Mohbadi/AppF1/main/Picture/unknown.jpg')
    
    
    # Affichage de la liste des pilotes
    selected_team = st.selectbox("Select a Team :", df_photo['name'])
    
    # Affichage de la photo du pilote sélectionné
    pilote_row = df_photo[df_photo['name'] == selected_team]
    pilote_photo_url = pilote_row.iloc[0]['Photo url']
    team = df_results.merge(df_constructor, on = 'constructorId').merge(df_races, on='raceId').sort_values(by='year', ascending=True)
    first_gp = sorted(team[(team["name_x"] == selected_team)]["year"].unique(), reverse=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(pilote_photo_url, width=370)
        st.markdown('<div style="border: 1px solid black; padding: 5px; text-align: center">Formula 1 Team</div>', unsafe_allow_html=True)
        inter = sorted(team[team['name_x'] == selected_team]["year"].unique(), reverse= False)
        start, end = st.select_slider(
        "Number of wins and points between :",
        options= inter,
        value= [inter[0],inter[-1]])
        m1, m2 = st.columns(2)
        m1.metric(":sports_medal: Wins :sports_medal:", calculate_historic_team_wins(start,end,selected_team) )
        m2.metric(":sports_medal: Points :sports_medal:", calculate_historic_team_point(start,end,selected_team) )


    data = {
    "Team name": [df_photo[df_photo['name']==selected_team]['name'].iloc[0]],
    "Nationality": [df_photo[df_photo['name']==selected_team]['nationality'].iloc[0]],
    "First Grand Prix": [first_gp[0]],
    "Grands Prix entered": [len(team[(team['name_x']==selected_team)]['raceId'].unique())],
    "Wins": [len(team[(team['name_x']==selected_team) & (team['positionOrder'] == 1)])],
    "Podium": [len(team[(team['name_x']==selected_team) & (team['positionOrder'] <= 3)])],
    "Points (Until last race)": [team[(team['name_x']==selected_team)]['points'].sum()],
    "Pole positions":[team[(team['name_x']==selected_team) & (team['grid'] == 1)]['grid'].sum()]
    }
    

    df_team = pd.DataFrame(data)

    # Transposer le DataFrame pour mettre les équipes sur les lignes
    df_team = df_team.T
    
    # Réinitialiser les index
    df_team.reset_index(inplace=True)
    
    # Renommer les colonnes
    df_team.columns = ["Category", "  _______________________  "]
    with col2:
        st.dataframe(df_team, 
             hide_index=True)
    
    with st.expander(f"Deeper statistics on {selected_team}"):
        start, end = st.select_slider(
        "Number of wins and points between :",
        options= inter,
        value= [inter[0],inter[-1]], key=["y","e"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>Race finished rate between {start} and {end}</h1></div>", unsafe_allow_html=True)
            plot_completion_team_rate(selected_team, start, end)
        with c2:
            st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>Podium finishes rate between {start} and {end}</h1></div>", unsafe_allow_html=True)
            plot_podium_team_percentage(selected_team, start, end)
        
        st.markdown('---')
        y = st.selectbox("Select a year :", sorted(team[team['name_x'] == selected_team]["year"].unique(), reverse= True), key="y")
        st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>{selected_team}'s position in {y} compared to the others</h1></div>", unsafe_allow_html=True)
        plot_team_positions_by_year(y, selected_team)
        
        st.markdown('---')
        y1 = st.selectbox("Select a year :", sorted(team[team['name_x'] == selected_team]["year"].unique(), reverse= True), key="y1")
        st.markdown(f"<div style='text-align: center;'><h1 style='font-size: 15px;'>{selected_team} mechanic crew performance during pit-stops in {y1} compared to the others</h1></div>", unsafe_allow_html=True)
        boxplot_by_team_for_year(y1, selected_team)
        
# Function for the circuits tab
def circuits_info():
    st.markdown("<div style='text-align: center;'><h1>Circuits</h1></div>", unsafe_allow_html=True)

    
    create_map()
    
    st.markdown('---')
    
    df_photo1 = df_circuit.merge(df_layout, on='location', how='left')
    df_photo1 = df_photo1.sort_values(by="Photo url")
    df_photo1['Photo url'] = df_photo1['Photo url'].fillna('https://raw.githubusercontent.com/Mohbadi/AppF1/main/Picture/unknown.jpg')
    
    # Affichage de la liste des pilotes
    selected_circuit = st.selectbox("Select a circuit :", df_photo1['name'])
    
    # Affichage de la photo du pilote sélectionné
    circuit_row = df_photo1[df_photo1['name'] == selected_circuit]
    circuit_photo_url = circuit_row.iloc[0]['Photo url']
    
    
    
    
    circuit = df_results.merge(df_races, on = "raceId").merge(df_circuit, on = 'circuitId').merge(df_drivers, on='driverId').sort_values(by='year', ascending=False)
    year_apperance = sorted(circuit[(circuit["name_y"] == selected_circuit)]["year"].unique(), reverse=True)
    first_gp = sorted(circuit[(circuit["name_y"] == selected_circuit)]["year"].unique(), reverse=False)
            
    data = {
    "Circuit name": [df_photo1[df_photo1['name']==selected_circuit]['name'].iloc[0]],
    "Country" : [df_photo1[df_photo1['name']==selected_circuit]['country'].iloc[0]],
    "Location" : [df_photo1[df_photo1['name']==selected_circuit]['location'].iloc[0]],
    "Altitude (m)" : [df_photo1[df_photo1['name']==selected_circuit]['alt'].iloc[0]],
    "Length (km)" : [df_photo1[df_photo1['name']==selected_circuit]['Length'].iloc[0]],
    "Turns" : [df_photo1[df_photo1['name']==selected_circuit]['Turns'].iloc[0]],
    "Number of Laps" : circuit[(circuit['name_y'] == selected_circuit) & (circuit['year'] == 2023) & (circuit['positionOrder'] == 1)]['laps'].iloc[0]
    }
    

    df_team = pd.DataFrame(data)

    # Transposer le DataFrame pour mettre les équipes sur les lignes
    df_team = df_team.T
    
    # Réinitialiser les index
    df_team.reset_index(inplace=True)
    
    # Renommer les colonnes
    df_team.columns = ["Category", "  _______________________  "]
    with st.expander(f"Deeper statistics on {selected_circuit}"):
        c1,c, c3, c2= st.columns(4)
        c.metric("Last Grand Prix", year_apperance[0])
        c1.metric("First Grand Prix", first_gp[0])
        c2.metric("Last winner", circuit[(circuit['name_y'] == selected_circuit) & (circuit['year'] == year_apperance[0]) & (circuit['positionOrder'] == 1)]['Driver_name'].iloc[0])
        c3.metric("Number of GP", len(year_apperance))
        
    col1, col2 = st.columns(2)    
    with col1:
        st.image(circuit_photo_url, width=380)
        
        st.markdown('<div style="border: 1px solid black; padding: 5px; text-align: center">Track layout</div>', unsafe_allow_html=True)
    
    with col2:
        st.dataframe(df_team, 
             hide_index=True)
    
    st.markdown('---')
    
    with st.expander("Analysis of the effects of track characteristics"):

        engine_failures_vs_altitude()
        
        st.markdown("---")
        
        st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Correlation between starting position and finishing position by circuit</h1></div>", unsafe_allow_html=True)

        # Fusionner les résultats avec les données des pilotes et des courses
        results_with_race = df_results.merge(df_races, on='raceId')
        circuit_name = df_circuit[['circuitId','circuitRef']]
        results_with_race = results_with_race.merge(circuit_name, on = 'circuitId')
        results_with_race = results_with_race[results_with_race['year'] >= 2021]
        
        # Calculer la corrélation entre la position de départ et la position d'arrivée pour chaque course
        correlation_by_race = results_with_race.groupby('circuitRef')[['grid', 'positionOrder']].corr().iloc[0::2, -1].sort_values(ascending=False).head(25)
        # Tracer le graphique à barres
        plt.figure(figsize=(12, 6))
        correlation_by_race.plot(kind='bar', color='grey')
        plt.xlabel('Circuit')
        plt.ylabel('Correlation')
        plt.xticks(rotation=45, ha='right', ticks=range(len(correlation_by_race)), labels=correlation_by_race.index.get_level_values(0))
        plt.grid(axis='y', alpha=0.2)
        st.pyplot()


def pred_podium():
    st.markdown("<div style='text-align: center;'><h1>Podium Prediction</h1></div>", unsafe_allow_html=True)

    with open('GB.pkl', 'rb') as f:
        model = pickle.load(f)

    
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Driver Input"):
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Driver card</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            driver_name = st.selectbox("Select a driver", df_drivers["Driver_name"])
            driverId = df_drivers[df_drivers["Driver_name"] == driver_name]["driverId"].values[0]
            # Liste des nationalités disponibles (à adapter selon votre dataset)
            nationality = ["American", "American-Italian","Argentine", "Argentine-Italian", "Australian", "Austrian", "Belgian", "Brazilian", "British", "Canadian", "Chilean", "Chinese", "Colombian", "Czech", "Danish", "Dutch", "East German", "Finnish",'French',
                                "German", "Hungarian", "Indian", "Indonesian", "Irish", "Italian", "Japanese", "Liechtensteiner", "Malaysian", "Mexican", "Monegasque", "New Zealander", "Polish","Portuguese","Rhodesian", "Russian", "South African", "Spanish","Swedish",
                                "Swiss", "Thai", "Uruguayan","Venezuelan"]
    
            # Créer une instance de OneHotEncoder
            encoder = OneHotEncoder(categories=[nationality], sparse_output=False, handle_unknown='ignore')
    
            # Fit l'encodeur sur les nationalités disponibles
            encoder.fit(np.array(nationality).reshape(-1, 1))
    
    
            # Sélection de la nationalité
            selected_nationality = st.selectbox('Sélectionnez une nationalité', nationality)
    
            # Transformation en dummy
            def transform_nationality(selected_nationality, encoder):
                transformed = encoder.transform(np.array([selected_nationality]).reshape(-1, 1))
                return transformed
            # Transformation de la nationalité sélectionnée en dummies
            dummy_nationality = transform_nationality(selected_nationality, encoder).flatten()
     
            c1,c2 = st.columns(2)
            with c1:
                
                age_race_start = st.number_input("Age at race start", 0, 100)
                starting_position = st.number_input("Starting position", 1, 20)
            with c2:
                
                exp = st.number_input("Number of race start", 0, 400)
                last_result = st.number_input("Last race result", 1, 20)
                
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Driver Performance</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            col_1,col_2 = st.columns(2)
            with col_1:
                driver_podium_rate_last_y = st.number_input("Driver Podium rate (Last Year)")
                avg_driver_pos_last_y = st.number_input("Avg Driver position (Last Year)")

            with col_2:
                driver_podium_rate_this_y = st.number_input("Driver podium rate (This Year until last race)")
                avg_driver_pos_this_y = st.number_input("Avg driver position (This Year until last race)")
            df_final1 = df_final.merge(df_drivers[["driverId", "Driver_name"]], on="driverId")
            yearofbirth = df_final1[df_final1["Driver_name"] == driver_name]["Yearofbirth"].iloc[0]
            
    with col2:
        with st.expander("Team Input"):
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Current Team</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                constructor_redbull = 9 if st.radio("Red Bull", ["Yes", "No"]) == "Yes" else 0
                constructor_ferrari = 6 if st.radio("Ferrari", ["Yes", "No"]) == "Yes" else 0
            with c2:
                constructor_mercedes = 131 if st.radio("Mercedes", ["Yes", "No"]) == "Yes" else 0
                constructor_mclaren = 1 if st.radio("Mclaren", ["Yes", "No"]) == "Yes" else 0
            
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Team performance</h1></div>", unsafe_allow_html=True)
            st.markdown("---")

            avg_pit_stop = st.number_input("Average Pit stop (seconds)")
            avg_team_pos_last_y = st.number_input("Avg Team position (Last Year)")
            avg_team_pos_this_y = st.number_input("Avg Team position (This Year until last race)")

    with st.expander("Race Input"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Formula 1 Event</h1></div>", unsafe_allow_html=True)
            st.markdown('---')
            raceId = 1121
            year = st.slider("Choose a year", 2023, 2024)
            round_num = st.number_input("Round number", 1, 24)
            circuit_name = st.selectbox("Select a circuit", df_circuit["name"])
            circuitId = df_circuit[df_circuit["name"] == circuit_name]["circuitId"].values[0]
            alt = df_circuit[df_circuit["name"] == circuit_name]["alt"].values[0]
        
        with c2:
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Weather during the race</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                weather_warm = 1 if st.radio("Warm", ["Yes", "No"]) == "Yes" else 0
                weather_cold = 1 if st.radio("Cold", ["Yes", "No"]) == "Yes" else 0
                weather_cloudy = 1 if st.radio("Cloudy", ["Yes", "No"]) == "Yes" else 0
            with c2:
                weather_dry = 1 if st.radio("Dry", ["Yes", "No"]) == "Yes" else 0
                weather_wet = 1 if st.radio("Wet", ["Yes", "No"]) == "Yes" else 0



    # Préparer toutes les autres variables comme des scalaires
    other_features = np.array([raceId, year, round_num, driverId, starting_position, circuitId, alt, avg_pit_stop,
                               driver_podium_rate_last_y, driver_podium_rate_this_y, last_result, avg_driver_pos_last_y,
                               avg_team_pos_last_y, avg_driver_pos_this_y, avg_team_pos_this_y, yearofbirth,
                               age_race_start, exp, weather_warm, weather_cold, weather_dry, weather_wet, weather_cloudy,
                               constructor_redbull, constructor_mclaren, constructor_ferrari, constructor_mercedes])

    # Concaténer les dummies de nationalité avec les autres caractéristiques
    row = np.concatenate([other_features, dummy_nationality])
    row = row.reshape(1, -1)
    scaler = joblib.load("scaler_GB.joblib")
    row1 = scaler.transform(row.reshape(1, -1))    
    prediction = model.predict(row1)
    st.write(row1)
    st.write(prediction)

    c1, c2, c3 = st.columns(3)
    
    with c1:
        pass
    with c2:
        
        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        if st.button('Prédire'):
            prediction = model.predict(row1)
            # Affichage du résultat de manière esthétique
            if prediction == 1:
                with st.info("Podium"):
                    st.markdown("<h1 style='text-align: center; color: green;'>Podium</h1>", unsafe_allow_html=True)
            else:
                with st.warning("No podium"):
                    st.markdown("<h1 style='text-align: center; color: red;'>No podium</h1>", unsafe_allow_html=True)
    with c3:
        pass

def pred_win():
    st.markdown("<div style='text-align: center;'><h1>Winner Prediction</h1></div>", unsafe_allow_html=True)

    # Charger votre modèle
    model_svm = joblib.load('SVM.joblib')

    
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Driver Input"):
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Driver card</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            driver_name = st.selectbox("Select a driver", df_drivers["Driver_name"])
            driverId = df_drivers[df_drivers["Driver_name"] == driver_name]["driverId"].values[0]
            # Liste des nationalités disponibles (à adapter selon votre dataset)
            nationality = ["American", "American-Italian","Argentine", "Argentine-Italian", "Australian", "Austrian", "Belgian", "Brazilian", "British", "Canadian", "Chilean", "Chinese", "Colombian", "Czech", "Danish", "Dutch", "East German", "Finnish",'French',
                                "German", "Hungarian", "Indian", "Indonesian", "Irish", "Italian", "Japanese", "Liechtensteiner", "Malaysian", "Mexican", "Monegasque", "New Zealander", "Polish","Portuguese","Rhodesian", "Russian", "South African", "Spanish","Swedish",
                                "Swiss", "Thai", "Uruguayan","Venezuelan"]
    
            # Créer une instance de OneHotEncoder
            encoder = OneHotEncoder(categories=[nationality], sparse_output=False, handle_unknown='ignore')
    
            # Fit l'encodeur sur les nationalités disponibles
            encoder.fit(np.array(nationality).reshape(-1, 1))
    
    
            # Sélection de la nationalité
            selected_nationality = st.selectbox('Sélectionnez une nationalité', nationality)
    
            # Transformation en dummy
            def transform_nationality(selected_nationality, encoder):
                transformed = encoder.transform(np.array([selected_nationality]).reshape(-1, 1))
                return transformed
            # Transformation de la nationalité sélectionnée en dummies
            dummy_nationality = transform_nationality(selected_nationality, encoder).flatten()
     
            c1,c2 = st.columns(2)
            with c1:
                
                age_race_start = st.number_input("Age at race start", 0, 100)
                starting_position = st.number_input("Starting position", 1, 20)
            with c2:
                
                exp = st.number_input("Number of race start", 0, 400)
                last_result = st.number_input("Last race result", 1, 20)
                
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Driver Performance</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            col_1,col_2 = st.columns(2)
            with col_1:
                driver_podium_rate_last_y = st.number_input("Driver Podium rate (Last Year)")
                avg_driver_pos_last_y = st.number_input("Avg Driver position (Last Year)")

            with col_2:
                driver_podium_rate_this_y = st.number_input("Driver podium rate (This Year until last race)")
                avg_driver_pos_this_y = st.number_input("Avg driver position (This Year until last race)")
            
    with col2:
        with st.expander("Team Input"):
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Current Team</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                constructor_redbull = 9 if st.radio("Red Bull", ["Yes", "No"]) == "Yes" else 0
                constructor_ferrari = 6 if st.radio("Ferrari", ["Yes", "No"]) == "Yes" else 0
            with c2:
                constructor_mercedes = 131 if st.radio("Mercedes", ["Yes", "No"]) == "Yes" else 0
                constructor_mclaren = 1 if st.radio("Mclaren", ["Yes", "No"]) == "Yes" else 0
            
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Team performance</h1></div>", unsafe_allow_html=True)
            st.markdown("---")

            avg_pit_stop = st.number_input("Average Pit stop (seconds)")
            avg_team_pos_last_y = st.number_input("Avg Team position (Last Year)")
            avg_team_pos_this_y = st.number_input("Avg Team position (This Year until last race)")

    with st.expander("Race Input"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Formula 1 Event</h1></div>", unsafe_allow_html=True)
            st.markdown('---')
            raceId = 1121
            year = st.slider("Choose a year", 2023, 2024)
            round_num = st.number_input("Round number", 1, 24)
            circuit_name = st.selectbox("Select a circuit", df_circuit["name"])
            circuitId = df_circuit[df_circuit["name"] == circuit_name]["circuitId"].values[0]
            alt = df_circuit[df_circuit["name"] == circuit_name]["alt"].values[0]
        
        with c2:
            st.markdown("<div style='text-align: center;'><h1 style='font-size: 17px;'>Weather during the race</h1></div>", unsafe_allow_html=True)
            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                weather_warm = 1 if st.radio("Warm", ["Yes", "No"]) == "Yes" else 0
                weather_cold = 1 if st.radio("Cold", ["Yes", "No"]) == "Yes" else 0
                weather_cloudy = 1 if st.radio("Cloudy", ["Yes", "No"]) == "Yes" else 0
            with c2:
                weather_dry = 1 if st.radio("Dry", ["Yes", "No"]) == "Yes" else 0
                weather_wet = 1 if st.radio("Wet", ["Yes", "No"]) == "Yes" else 0
                




    # Préparer toutes les autres variables comme des scalaires
    other_features = np.array([raceId, year, round_num, driverId, starting_position, circuitId, alt, avg_pit_stop,
                               driver_podium_rate_last_y, driver_podium_rate_this_y, last_result, avg_driver_pos_last_y,
                               avg_team_pos_last_y, avg_driver_pos_this_y, avg_team_pos_this_y,
                               age_race_start, exp, weather_warm, weather_cold, weather_dry, weather_wet, weather_cloudy,
                               constructor_redbull, constructor_mclaren, constructor_ferrari, constructor_mercedes])

    # Concaténer les dummies de nationalité avec les autres caractéristiques
    row = np.concatenate([other_features, dummy_nationality])
    row = row.reshape(1, -1)
    
    scaler = joblib.load("scaler_svm.joblib")
    row1 = scaler.transform(row.reshape(1, -1))    
    prediction = model_svm.predict(row1)
    st.write(row1)
    st.write(prediction)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        pass
    with c2:
        
        st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
        if st.button('Prédire'):
            prediction = model_svm.predict(row1)
            # Affichage du résultat de manière esthétique
            if prediction == 1:
                with st.info("Podium"):
                    st.markdown("<h1 style='text-align: center; color: green;'>Win</h1>", unsafe_allow_html=True)
            else:
                with st.warning("No podium"):
                    st.markdown("<h1 style='text-align: center; color: red;'>Lose</h1>", unsafe_allow_html=True)
    with c3:
        pass







# Centered title in the sidebar with enhanced appearance
st.sidebar.image('F1Logo.png', use_column_width=True)
st.sidebar.markdown(
    "<div style='text-align: center;'><h1><strong>Formula 1 Decision Maker</strong></h1></div>", 
    unsafe_allow_html=True
)
st.sidebar.markdown('---')
st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Navigation</h1>", unsafe_allow_html=True)

# Display buttons for navigation
selected_page = st.sidebar.radio("",["Home", "Formula 1's Season", "Driver Stats", "Constructor Stats", "Circuits", "Podium Finish Prediction", "Winner Prediction"])

if selected_page == "Home":
    home()
elif selected_page == "Formula 1's Season":
    current_season()
elif selected_page == "Driver Stats":
    driver_stats()
elif selected_page == "Constructor Stats":
    constructor_stats()
elif selected_page == "Circuits":
    circuits_info()
elif selected_page == "Podium Finish Prediction":
    pred_podium()
elif selected_page == "Winner Prediction":
    pred_win()
    
st.sidebar.markdown('---')
st.sidebar.markdown(
    f"<div style='text-align: center;'><h1><strong>Mohamed & Antonin</strong></h1></div>", 
    unsafe_allow_html=True
)
