#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import time
import requests, json
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim  # geopy is a Python client for several popular geocoding web services
from geopy.exc import GeocoderTimedOut
import folium
from sklearn.cluster import KMeans

print("Folium version = ", folium.__version__)

# # City1 : Cairo/Egypt
web_Cai = requests.get("https://en.wikipedia.org/wiki/Category:Districts_of_Cairo").text
web_Cai

soup_Cai = BeautifulSoup(web_Cai)

soup_Cai

tag_Cai = soup_Cai.find(class_="mw-category")

tag_Cai

li_Cai = tag_Cai.find_all("li")
li_Cai

li_Cai.pop(0)

li_Cai

li_Cai.pop(0)

# li_Cai.pop(0)
Cai_neighbourhoods = []
for li_Cais in li_Cai:
    neighbourhood = li_Cais.text.split(',')[0]
    neighbourhood = neighbourhood.split('(')[0]
    Cai_neighbourhoods.append(neighbourhood)

# Cai_neighbourhoods=Cai_neighbourhoods.pop(30)
Cairo_neighborhood = list(set(Cai_neighbourhoods))
Cairo_neighborhood

df_Cairo = pd.DataFrame(Cairo_neighborhood, columns=['Neighbourhood'])
df_Cairo['City'] = 'Cairo'
df_Cairo['Country'] = 'Egypt'

df_Cairo.head()

# # City 2 :Istanbul/Turkey

web_Ist = requests.get("https://en.wikipedia.org/wiki/List_of_districts_of_Istanbul").text
web_Ist

soup_Ist = BeautifulSoup(web_Ist)
soup_Ist

tag_Ist = soup_Ist.find("table", class_="wikitable sortable")
tag_Ist

tr_Ist = tag_Ist.find_all("tr")
tr_Ist

td_Ist = tr_Ist[1:]
td_Ist

Ist_neighbourhoods = []
for td_Ist in tr_Ist[1:-4]:
    Ist_neighbourhoods.append(td_Ist.text.split("\n")[1])

Ist_neighbourhoods

df_Istanbul = pd.DataFrame(Ist_neighbourhoods, columns=['Neighbourhood'])
df_Istanbul['City'] = 'Istanbul'
df_Istanbul['Country'] = 'Turkey'

df_Istanbul.head()

# # City 3: Marrakesh/Morroco

web_Mar = requests.get("https://en.wikipedia.org/wiki/Subdivisions_of_Marrakesh").text
web_Mar

soup_Mar = BeautifulSoup(web_Mar)
soup_Mar

tag_Mar = soup_Mar.find("table", class_="wikitable sortable")
tag_Mar

tr_Mar = tag_Mar.find_all("tr")[1:]

tr_Mar

Mar_neighbourhoods = []
for td_Mar in tr_Mar:
    td_Mars = td_Mar.find_all("td")
    Mar_neighbourhoods.append(td_Mars[0].text)

Mar_neighbourhoods

df_Marrakesh = pd.DataFrame(Mar_neighbourhoods, columns=['Neighbourhood'])
df_Marrakesh['City'] = 'Marrakesh'
df_Marrakesh['Country'] = 'Morocco'

df_Marrakesh.head()

df_All = pd.concat([df_Cairo, df_Istanbul, df_Marrakesh], ignore_index=True)
df_All

df_All.to_excel("Neighbourhoods_Ca_Ist_Mar.xlsx", index=False)

df = pd.read_excel("Neighbourhoods_Ca_Ist_Mar.xlsx")

df.iterrows

x = []
for index, data in df.iterrows():
    # print(data['Neighbourhood'] + ', ' + data['City'])
    # print(index)
    # print(data)
    address = data['Neighbourhood'] + ', ' + data['City']
    geolocator = Nominatim(user_agent='Markov_Applied_DS')
    location = geolocator.geocode(address)
    x.append(location.latitude)
    print(x)

latitude = []
longitude = []

for index, data in df.iterrows():
    address = data['Neighbourhood'] + ', ' + data['City']

    try:
        geolocator = Nominatim(user_agent='Markov_Applied_DS')
        location = geolocator.geocode(address)
    except GeocoderTimedOut as err:
        print(err)
        latitude.append('timeout')
        longitude.append('timeout')
    else:
        if location is not None:
            latitude.append(location.latitude)
            longitude.append(location.longitude)
            print('The geograpical coordinate of {} are {}, {}.'.format(address, location.latitude, location.longitude))
        else:
            latitude.append(None)
            longitude.append(None)
            print('None Coordinates for {}'.format(address))
        time.sleep(1.0)

print("=================================")
print("=================================")
print("Total Latitude: ", len(latitude))
print("Total Longitude: ", len(longitude))

df['Latitude'] = latitude
df['Longitude'] = longitude

df

df['Latitude'].isnull().sum()

df.to_excel("Neighbourhoods_Ca_Ist_Mar_v1.xlsx", index=False)

df = pd.read_excel("Neighbourhoods_Ca_Ist_Mar_v1.xlsx")
df.isna().sum()

# # **4. Drawing Neighbourhoods/Cities Maps**

df_map = pd.read_excel("Neighbourhoods_Ca_Ist_Mar_v1.xlsx")

df_map

df_map.groupby('City').count()

city_loca = []
for city in list(df_map.groupby('City').count().index):
    geolocator = Nominatim(user_agent='Markov_Applied_DS')
    loca = geolocator.geocode(city)
    city_loca.append([city, loca.longitude, loca.latitude])
    # time.sleep(2.0)

city_loca

df_city = pd.DataFrame(data=city_loca, columns=['City', 'Longitude', 'Latitude'])
df_city

city_map = folium.Map(location=[0, 0], zoom_start=2)

# add markers to the cities
for lat, lng, label in zip(df_city['Latitude'], df_city['Longitude'], df_city['City']):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(city_map)

city_map

location = [df_city.loc[df_city['City'] == "Istanbul", 'Latitude'].values[0],
            df_city.loc[df_city['City'] == "Istanbul", 'Longitude'].values[0]]

location


# ## Neighbourhoods Maps

def generate_map(city, zoom):
    map = folium.Map(location=[df_city.loc[df_city['City'] == city, 'Latitude'].values[0],
                               df_city.loc[df_city['City'] == city, 'Longitude'].values[0]], zoom_start=zoom)
    # add markers to the cities
    for lat, lng, label in zip(df_map.loc[df_map['City'] == city, 'Latitude'],
                               df_map.loc[df_map['City'] == city, 'Longitude'],
                               df_map.loc[df_map['City'] == city, 'Neighbourhood']):
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7).add_to(map)

    return map


map_Cairo = generate_map('Cairo', 11)
map_Cairo

map_Istanbul = generate_map('Istanbul', 11)
map_Istanbul

map_Marrakesh = generate_map('Marrakesh', 11)
map_Marrakesh

# # 5. Foursquare Location Services API

# ### Getting Venues from Foursquare

test_lat = '30.0492'
test_lng = '30.9762'
test_radius = 1000

url = 'https://api.foursquare.com/v3/places/search?&client_id={}&client_secret={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, CLIENT_SECRET, test_lat, test_lng, test_radius, LIMIT)

headers = {
    "Accept": "application/json",
    "Authorization": "fsq3rngNKjsXko2yiCDm2rINmspp7G4ZpMf2YHdkN+8sdyA="
}

response = requests.request("GET", url, headers=headers)

print(url)

response = requests.request("GET", url, headers=headers)
response.json()

headers = {
    "Accept": "application/json",
    "Authorization": "fsq3rngNKjsXko2yiCDm2rINmspp7G4ZpMf2YHdkN+8sdyA="
}


def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    venues_list = []
    for name, lat, lng in zip(names, latitudes, longitudes):
        # create the API request URL
        url = 'https://api.foursquare.com/v3/places/search?&client_id={}&client_secret={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            lat,
            lng,  ############### V3
            radius,
            LIMIT)

        response = requests.request("GET", url, headers=headers)  ####### V3
        response = response.json()['results']

        if (len(response) == 0):
            print('0')

        # return only relevant information for each nearby venue 'results'][1]['categories'][0]['name'] 
        for v in response:

            if (v['categories']):
                venues_list.append([(
                    name,
                    lat,
                    lng,
                    v['name'],
                    v['geocodes']['main']['latitude'],
                    v['geocodes']['main']['longitude'],
                    v['categories'][0]['name'])])

    print("Number of Neighbourhoods at the City ", len(venues_list))

    nearby_venues = pd.DataFrame([venue for neigh in venues_list for venue in neigh])
    # venues_list : list of lists of tubles "list of neighbourhoods, of list of venues, of tuble of venue data"
    nearby_venues.columns = ['Neighbourhood',
                             'Neighbourhood Latitude',
                             'Neighbourhood Longitude',
                             'Venue',
                             'Venue Latitude',
                             'Venue Longitude',
                             'Venue Category']

    return (nearby_venues)


# ##  Cairo Venues

df_Caivenues = getNearbyVenues(df_map.loc[df_map['City'] == 'Cairo', 'Neighbourhood'],
                               df_map.loc[df_map['City'] == 'Cairo', 'Latitude'],
                               df_map.loc[df_map['City'] == 'Cairo', 'Longitude'], 1000)

df_Caivenues

# ##  Istanbul Venues

df_Istvenues = getNearbyVenues(df_map.loc[df_map['City'] == 'Istanbul', 'Neighbourhood'],
                               df_map.loc[df_map['City'] == 'Istanbul', 'Latitude'],
                               df_map.loc[df_map['City'] == 'Istanbul', 'Longitude'], 1000)

df_Istvenues

# ## Marrakesh Venues


df_Marvenues = getNearbyVenues(df_map.loc[df_map['City'] == 'Marrakesh', 'Neighbourhood'],
                               df_map.loc[df_map['City'] == 'Marrakesh', 'Latitude'],
                               df_map.loc[df_map['City'] == 'Marrakesh', 'Longitude'], 1000)

df_Marvenues

# ###  Save output DataFrames in one Excel File


# save dataframes to an excel file
with pd.ExcelWriter('Neighbourhoods_Cai_Ist_Mar_with_Venues.xlsx') as writer:
    df_Caivenues.to_excel(writer, sheet_name='Caivenues', index=False)
    df_Istvenues.to_excel(writer, sheet_name='Istvenues', index=False)
    df_Marvenues.to_excel(writer, sheet_name='Marvenues', index=False)


# ## Most common Venue Categories at each Neighbourhood

# one hot encoding for venues categories
def onehot_df(df_venues):
    df_onehot = pd.get_dummies(df_venues[['Venue Category']], prefix="", prefix_sep="")
    df_onehot['Neighbourhood'] = df_venues['Neighbourhood']
    # move neighborhood column to the first column
    fixed_columns = [df_onehot.columns[-1]] + list(df_onehot.columns[:-1])
    df_onehot = df_onehot[fixed_columns]

    return df_onehot


# most common venues per neighboarhood
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


# sort top 20 venue categories
def sorted_dataframe(df_grouped):
    num_top_venues = 20

    indicators = ['st', 'nd', 'rd']

    # create dynamic list with columns' names according to number of top venues
    columns = ['Neighbourhood']
    for ind in range(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind + 1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind + 1))

    # create a new dataframe
    df_venues_sorted = pd.DataFrame(columns=columns)
    df_venues_sorted['Neighbourhood'] = df_grouped['Neighbourhood']

    for ind in range(df_grouped.shape[0]):  # number of neighbourhoods
        df_venues_sorted.iloc[ind, 1:] = return_most_common_venues(df_grouped.iloc[ind, :], num_top_venues)

    return df_venues_sorted


# ### Cairo Venues


df_Caivenues = pd.read_excel("Neighbourhoods_Cai_Ist_Mar_with_Venues.xlsx", sheet_name="Caivenues")
df_Caivenues.head()

df_Caivenues.shape

Cai_onehot = onehot_df(df_Caivenues)
Cai_onehot.head()

Cai_grouped = Cai_onehot.groupby('Neighbourhood').sum().reset_index()
Cai_grouped.head()

Cai_venues_sorted = sorted_dataframe(Cai_grouped)
Cai_venues_sorted.head()

# ### Istanbul Venues

df_Istvenues = pd.read_excel("Neighbourhoods_Cai_Ist_Mar_with_Venues.xlsx", sheet_name="Istvenues")
df_Istvenues.head()

df_Istvenues.shape

Ist_onehot = onehot_df(df_Istvenues)
Ist_onehot.head()

Ist_grouped = Ist_onehot.groupby('Neighbourhood').sum().reset_index()
Ist_grouped.head()

Ist_venues_sorted = sorted_dataframe(Ist_grouped)
Ist_venues_sorted.head()

# ### Marrakesh Venues

df_Marvenues = pd.read_excel("Neighbourhoods_Cai_Ist_Mar_with_Venues.xlsx", sheet_name="Marvenues")
df_Marvenues.head()

df_Marvenues.shape

Mar_onehot = onehot_df(df_Marvenues)
Mar_onehot.head()

Mar_grouped = Mar_onehot.groupby('Neighbourhood').sum().reset_index()
Mar_grouped.head()

Mar_venues_sorted = sorted_dataframe(Mar_grouped)
Mar_venues_sorted.head()

# save one hot venues dataframes to an excel file
with pd.ExcelWriter('Neighbourhoods_onehot_Venues_Categories.xlsx') as writer:
    Cai_grouped.to_excel(writer, sheet_name='Caivenues', index=False)
    Ist_grouped.to_excel(writer, sheet_name='Istvenues', index=False)
    Mar_grouped.to_excel(writer, sheet_name='Marvenues', index=False)

# save most common venues dataframes to an excel file
with pd.ExcelWriter('Neighbourhoods_most_common_Venues_Categories.xlsx') as writer:
    Cai_venues_sorted.to_excel(writer, sheet_name='Caivenues', index=False)
    Ist_venues_sorted.to_excel(writer, sheet_name='Istvenues', index=False)
    Mar_venues_sorted.to_excel(writer, sheet_name='Marvenues', index=False)


# # ** Individual Clustering**

def map_city_clusters(df_merged, longitude, latitude):
    # create map
    map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

    colors = ['#ff0000', '#8000ff', '#3c68f9', '#d4dd80', '#f6be68', '#ff964f', '#2adddd', '#b2f396']

    # add markers to the map
    for lat, lon, neigh, cluster in zip(df_merged['Latitude'], df_merged['Longitude'], df_merged['Neighbourhood'],
                                        df_merged['Cluster Labels']):
        label = str(neigh) + ' (Cluster ' + str(cluster) + ')'
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup=label,
            color=colors[cluster],
            fill=True,
            fill_color=colors[cluster],
            fill_opacity=0.7).add_to(map_clusters)

    return map_clusters


df_neigh = pd.read_excel("Neighbourhoods_Ca_Ist_Mar_v1.xlsx")
df_neigh

Cai_grouped = pd.read_excel("Neighbourhoods_onehot_Venues_Categories.xlsx", sheet_name="Caivenues")
Cai_grouped

# set number of clusters
kclusters = 8

Cai_grouped_clustering = Cai_grouped.drop('Neighbourhood', axis=1)

# run k-means clustering
kmeans_Cai = KMeans(n_clusters=kclusters, random_state=0).fit(Cai_grouped_clustering)

# check cluster labels generated for each row in the dataframe
len(kmeans_Cai.labels_)

Cai_merged = df_neigh.loc[df_neigh['City'] == 'Cairo']
Cai_merged = Cai_merged[Cai_merged['Neighbourhood'].isin(Cai_grouped['Neighbourhood'])].reset_index(drop=True)
Cai_merged

# add clustering labels
Cai_merged['Cluster Labels'] = kmeans_Cai.labels_

# merge Cai_merged with Cai_venues_sorted to add latitude/longitude for each neighborhood
Cai_merged = Cai_merged.join(Cai_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

Cai_merged.head()  # check the last columns!

Cai_merged.shape

df_city.loc[df_city['City'] == 'Cairo', 'Longitude']

Cai_lng = df_city.loc[df_city['City'] == 'Cairo', 'Longitude'][0]
Cai_lat = df_city.loc[df_city['City'] == 'Cairo', 'Latitude'][0]

map_city_clusters(Cai_merged, Cai_lng, Cai_lat)

Cai_merged.groupby('Cluster Labels').count()

# ## Cairo Clusters

Cai_merged.loc[Cai_merged['Cluster Labels'] == 0, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

Cai_merged.loc[Cai_merged['Cluster Labels'] == 1, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

Cai_merged.loc[Cai_merged['Cluster Labels'] == 2, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

Cai_merged.loc[Cai_merged['Cluster Labels'] == 3, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

Cai_merged.loc[Cai_merged['Cluster Labels'] == 4, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

Cai_merged.loc[Cai_merged['Cluster Labels'] == 5, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

Cai_merged.loc[Cai_merged['Cluster Labels'] == 6, Cai_merged.columns[[0] + list(range(6, Cai_merged.shape[1]))]].head()

# ## Istanbul Map

Ist_grouped = pd.read_excel("Neighbourhoods_onehot_Venues_Categories.xlsx", sheet_name="Istvenues")
Ist_grouped

# set number of clusters
kclusters = 8

Ist_grouped_clustering = Ist_grouped.drop('Neighbourhood', axis=1)

# run k-means clustering
kmeans_Ist = KMeans(n_clusters=kclusters, random_state=0).fit(Ist_grouped_clustering)

# check cluster labels generated for each row in the dataframe
len(kmeans_Ist.labels_)

Ist_merged = df_neigh.loc[df_neigh['City'] == 'Istanbul']
Ist_merged = Ist_merged[Ist_merged['Neighbourhood'].isin(Ist_grouped['Neighbourhood'])].reset_index(drop=True)
Ist_merged

# add clustering labels
Ist_merged['Cluster Labels'] = kmeans_Ist.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Ist_merged = Ist_merged.join(Ist_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

Ist_merged.head()  # check the last columns!

Ist_merged.shape

df_city.loc[df_city['City'] == 'Istanbul', 'Longitude']

Ist_lng = df_city.loc[df_city['City'] == 'Istanbul', 'Longitude'][1]
Ist_lat = df_city.loc[df_city['City'] == 'Istanbul', 'Latitude'][1]

map_city_clusters(Ist_merged, Ist_lng, Ist_lat)

Ist_merged.groupby('Cluster Labels').count()

# ## Istanbul Clusters

Ist_merged.loc[Ist_merged['Cluster Labels'] == 0, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 1, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 2, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 3, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 4, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 5, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 6, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

Ist_merged.loc[Ist_merged['Cluster Labels'] == 7, Ist_merged.columns[[0] + list(range(6, Ist_merged.shape[1]))]].head()

# ## Marrakesh Map

Mar_grouped = pd.read_excel("Neighbourhoods_onehot_Venues_Categories.xlsx", sheet_name="Marvenues")
Mar_grouped

# set number of clusters
kclusters = 8

Mar_grouped_clustering = Mar_grouped.drop('Neighbourhood', axis=1)

# run k-means clustering
kmeans_Mar = KMeans(n_clusters=kclusters, random_state=0).fit(Mar_grouped_clustering)

# check cluster labels generated for each row in the dataframe
len(kmeans_Mar.labels_)

Mar_merged = df_neigh.loc[df_neigh['City'] == 'Marrakesh']
Mar_merged = Mar_merged[Mar_merged['Neighbourhood'].isin(Mar_grouped['Neighbourhood'])].reset_index(drop=True)
Mar_merged

# add clustering labels
Mar_merged['Cluster Labels'] = kmeans_Mar.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Mar_merged = Mar_merged.join(Mar_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

Mar_merged.head()  # check the last columns!

Mar_merged.shape

df_city.loc[df_city['City'] == 'Marrakesh', 'Longitude']

Mar_lng = df_city.loc[df_city['City'] == 'Marrakesh', 'Longitude'][2]
Mar_lat = df_city.loc[df_city['City'] == 'Marrakesh', 'Latitude'][2]

map_city_clusters(Mar_merged, Mar_lng, Mar_lat)

Mar_merged.groupby('Cluster Labels').count()

# ## Marrakesh Clusters

Mar_merged.loc[Mar_merged['Cluster Labels'] == 0, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 1, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 2, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 3, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 4, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 5, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 6, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

Mar_merged.loc[Mar_merged['Cluster Labels'] == 7, Mar_merged.columns[[0] + list(range(6, Mar_merged.shape[1]))]].head()

# ## Save output DataFrames in one Excel File

# save dataframes to an excel file
with pd.ExcelWriter('Neighbourhoods_Clusters_with_Venues_Categories.xlsx') as writer:
    Cai_merged.to_excel(writer, sheet_name='Cai_clusters', index=False)
    Ist_merged.to_excel(writer, sheet_name='Ist_clusters', index=False)
    Mar_merged.to_excel(writer, sheet_name='Mar_clusters', index=False)


# # 7. Complete Clustering
 
def onehot_comp_df(df_venues):
    df_onehot = pd.get_dummies(df_venues[['Venue Category']], prefix="", prefix_sep="")
    df_onehot['Neighbourhood'] = df_venues['Neighbourhood']
    df_onehot['City'] = df_venues['City']
    # move neighborhood column to the first column
    fixed_columns = [df_onehot.columns[-1]] + [df_onehot.columns[-2]] + list(df_onehot.columns[:-2])
    df_onehot = df_onehot[fixed_columns]

    return df_onehot


def return_most_common_venues_comp(row, num_top_venues):
    row_categories = row.iloc[2:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


def sorted_comp_dataframe(df_grouped):
    num_top_venues = 20

    indicators = ['st', 'nd', 'rd']

    # create columns according to number of top venues
    columns = ['Neighbourhood', 'City']
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind + 1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind + 1))

    # create a new dataframe
    df_venues_sorted = pd.DataFrame(columns=columns)
    df_venues_sorted['Neighbourhood'] = df_grouped['Neighbourhood']
    df_venues_sorted['City'] = df_grouped['City']
    for ind in np.arange(df_grouped.shape[0]):
        df_venues_sorted.iloc[ind, 2:] = return_most_common_venues_comp(df_grouped.iloc[ind, :], num_top_venues)

    return df_venues_sorted


# import data
df_Caivenues = pd.read_excel("Neighbourhoods_with_Venues.xlsx", sheet_name="Caivenues")
df_Istvenues = pd.read_excel("Neighbourhoods_with_Venues.xlsx", sheet_name="Istvenues")
df_Marvenues = pd.read_excel("Neighbourhoods_with_Venues.xlsx", sheet_name="Marvenues")

df_Caivenues['City'] = 'Cairo'
df_Istvenues['City'] = 'Istanbul'
df_Marvenues['City'] = 'Marrakesh'
df_Caivenues

comp_venues = pd.concat([df_Caivenues, df_Istvenues, df_Marvenues], ignore_index=True)
comp_venues.shape

comp_venues

comp_onehot = onehot_comp_df(comp_venues)
comp_onehot

comp_grouped = comp_onehot.groupby(['Neighbourhood', 'City']).sum().reset_index()
comp_grouped

comp_venues_sorted = sorted_comp_dataframe(comp_grouped)
comp_venues_sorted

# set number of clusters
kclusters = 8

comp_grouped_clustering = comp_grouped.drop(['Neighbourhood', 'City'], axis=1)

# run k-means clustering
kmeans_comp = KMeans(n_clusters=kclusters, random_state=0).fit(comp_grouped_clustering)

# check cluster labels generated for each row in the dataframe
len(kmeans_comp.labels_)

comp_merged = df_neigh.copy()
comp_merged = comp_merged[comp_merged['Neighbourhood'].isin(comp_grouped['Neighbourhood'])].reset_index(drop=True)
comp_merged.shape

# add clustering labels
comp_merged['Cluster Labels'] = kmeans_comp.labels_

comp_merged = comp_merged.join(comp_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood', lsuffix='',
                               rsuffix='_right')

comp_merged.head()

map_city_clusters(comp_merged, Cai_lng, Cai_lat)

map_city_clusters(comp_merged, Ist_lng, Ist_lat)

map_city_clusters(comp_merged, Mar_lng, Mar_lat)

comp_merged.groupby(['Cluster Labels', 'City']).count()

comp_merged.loc[comp_merged['Cluster Labels'] == 0, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 1, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 2, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 3, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 4, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 5, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 6, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]

comp_merged.loc[comp_merged['Cluster Labels'] == 7, comp_merged.columns[[0] + list(range(6, comp_merged.shape[1]))]]
