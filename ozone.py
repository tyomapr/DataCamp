import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

dataset = pd.read_csv('data/ozone.csv')
df = dataset.copy()
df.columns = df.columns.str.strip()

null_counts = df.isnull().sum()

null_df = pd.DataFrame({
    'Column': null_counts.index,
    'DF Types': df.dtypes[null_counts.index].values,
    'Null Count': null_counts.values
})

null_df['Null Percentage'] = ((null_df['Null Count'] / len(df)) * 100).map("{:.2f}%".format)

print('The Missing Data Table:')
print(null_df)
print()
print(f'The dataset contains the following number of rows and columns overall: {df.shape}')
print(f'The dataset contains the potential number of duplicates: {df.duplicated().sum()}')

print("The messy dataset:")
df.head()

# As we can see, the column date is quite messy. Let's assume that within the first 10 rows we can detect and partly fix dateformats. We will start by 
# checking whether the dataset contains only 2024 year, the next target is the whole dateformat.

df.sort_values(by = ['County', 'Local Site Name'], ascending = True, inplace = True)
df.reset_index(drop = True, inplace = True)

df['Year'] = df['Date'].astype(str).str.extract(r'(\d{4})')
df = df[df['Year'].notna()]
unique_years = df['Year'].unique()
print(f'The unique year in dataset is: {unique_years}')

## The identified and unique of the dataset is 2024. The function data_check will change the format to "%m/%d/%Y" and the following df['Date'].isna().sum() will conclude whether there are any nulls left afterwards. The next step is to fix the dateformat as much as possible by replacing the missing values where the specific row of the column ['Date'] (where it is 2024 only) is located between two distinct dates. For instance, the row with 2024 only can be restored if it is between 01/01/2024 and 01/03/2024. We can logically conclude that in this case scenario, the date between them is 01/02/2024.

def data_check(x):
    x = str(x).strip()

    try:
        datetime.strptime(x, "%m/%d/%Y")
        return x
    except:
        pass

    try:
        dt = datetime.strptime(x, "%B %d/%Y")
        return dt.strftime("%m/%d/%Y")
    except:
        pass

    try:
        if x.startswith("/"):
            return x[1:]
    except:
        pass

    return np.nan

df['Date'] = df['Date'].apply(data_check)

print(f"Number of rows with failed date parsing: {df['Date'].isna().sum()}") # checking whether some of the days were not parsed

df['Interpolarity'] = 'Not Interpolated' 

# Restoring the days where the year 2024 only is present

i = 1
while i < len(df) - 1:
    if str(df.loc[i, 'Date']).strip() == '2024':
        counter = 1
        while i + counter < len(df) and str(df.loc[i + counter, 'Date']).strip() == '2024':
            counter += 1
        prev_date = pd.to_datetime(df.loc[i - 1, 'Date'], errors = 'coerce')
        next_date = pd.to_datetime(df.loc[i + counter, 'Date'], errors = 'coerce')
        if pd.notna(prev_date) and pd.notna(next_date) and (next_date - prev_date).days == counter + 1:
            for j in range(counter):
                df.loc[i + j, 'Interpolarity'] = 'Interpolated'
                interpolated_date = prev_date + pd.Timedelta(days = j + 1)
                df.loc[i + j, 'Date'] = interpolated_date.strftime('%m/%d/%Y')
            i += counter
        else:
            i += counter
    else:
        i += 1


# Dropping the nulls of the most important columns and filling in nulls, finding weekends and weekdays

df = df[df['Date'] != '2024']
df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')
df = df.sort_values(by=['County', 'Local Site Name', 'Date'])
df.reset_index(drop = True, inplace = True)
df['Day Of Week'] = df['Date'].dt.dayofweek
df['Type Of Day'] = df['Day Of Week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
df['Date'] = df['Date'].dt.strftime('%Y/%m/%d')

df = df.dropna(subset = {'Daily Max 8-hour Ozone Concentration', 'Daily AQI Value'})
df = df.drop(columns = ['Year', 'Interpolarity', 'Day Of Week'], axis = 1)
df = df.fillna('unknown')
print(f"Number of deleted duplicated rows are: {df.duplicated().sum()}")
print()
df = df.drop_duplicates()
df.head()
df.info() # The final result of cleaning the data
unique_counts = df.nunique()
pd.DataFrame({'Column': unique_counts.index, 'Unique Count': unique_counts.values})
df.head()

days = df.copy()
amount_of_days = days.groupby('Local Site Name')['Date'].count().to_frame(name = 'Amount Of Days In Data')
print('The amount of days for each local site name:')
display(amount_of_days)

# question 2

import seaborn as sns
import matplotlib.pyplot as plt

df_filtered = df.select_dtypes(include = 'number').drop(columns = ['Site ID', 'POC', 'County FIPS Code'], errors = 'ignore')
numeric_df = df_filtered
corr = numeric_df.corr()
plt.figure(figsize = (10, 8))
sns.heatmap(corr, annot = True, fmt = ".2f", cmap = 'coolwarm', square = True, linewidths = 0.5, cbar_kws = {"shrink": 0.75})
plt.title("Correlation Matrix", fontsize = 14)
plt.xticks(rotation = 45, ha = 'right')
plt.yticks(rotation = 0)
plt.tight_layout()
plt.show()

grouped = df.groupby('County')[['Daily Max 8-hour Ozone Concentration', 'Daily AQI Value']].mean()
grouped = grouped.sort_values(by = 'Daily Max 8-hour Ozone Concentration', ascending = True)
fig, ax1 = plt.subplots(figsize = (12, 14))
counties = grouped.index
ozone = grouped['Daily Max 8-hour Ozone Concentration']
aqi = grouped['Daily AQI Value']
y_pos = np.arange(len(counties))
ax1.barh(y_pos - 0.2, ozone, height = 0.4, color = 'skyblue', label = 'Ozone Concentration')
ax1.set_xlabel('Ozone Concentration')
ax1.set_ylabel('County')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(counties)
ax1.invert_yaxis()
ax2 = ax1.twiny()
ax2.barh(y_pos + 0.2, aqi, height = 0.4, color = 'orange', label = 'AQI Value')
ax2.set_xlabel('AQI Value')
plt.title('Average Ozone Concentration and Daily AQI by County')
fig.tight_layout()
blue_patch = plt.Line2D([0], [0], color = 'skyblue', lw = 4, label = 'Ozone Concentration')
orange_patch = plt.Line2D([0], [0], color = 'orange', lw = 4, label = 'AQI Value')
ax1.legend(handles = [blue_patch, orange_patch], loc = 'upper right')
plt.show()

df_copy = df.copy()
df_copy = df_copy.drop(['Date', 'Site ID', 'POC', 'County FIPS Code'], axis = 1)
display(df_copy.describe())

# part 3

# Creating three visualizations in order to see the difference over time and regions using top and bottom counties in order not to overwhelm the visualizations

import plotly.express as px
import plotly.graph_objects as go

top_counties = df['County'].value_counts().head(5).index
bottom_counties = df['County'].value_counts().tail(5).index
df['County Group'] = df['County']
df.loc[df['County'].isin(top_counties), 'County Group'] = df['County'] + ' (Top 5)'
df.loc[df['County'].isin(bottom_counties), 'County Group'] = df['County'] + ' (Bottom 5)'
subset = df[df['County'].isin(top_counties.union(bottom_counties))]
subset['Month'] = pd.to_datetime(subset['Date']).dt.to_period('M').dt.to_timestamp()
monthly_avg = subset.groupby(['Month', 'County Group'])['Daily Max 8-hour Ozone Concentration'].mean().reset_index()

fig = px.line(
    monthly_avg,
    x = 'Month',
    y = 'Daily Max 8-hour Ozone Concentration',
    color = 'County Group',
    title = 'Monthly Average Ozone Concentration (Top 5 vs Bottom 5 Counties)'
)

fig.update_layout(
    xaxis_title = "Month",
    yaxis_title = "Avg Ozone Concentration (ppm)",
    legend_title = "County Group"
)

fig.show()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by = 'Date')


county_avg = df.groupby('County')['Daily Max 8-hour Ozone Concentration'].mean()
top_5 = county_avg.sort_values(ascending = False).head(5).index
bottom_5 = county_avg.sort_values().head(5).index
selected_counties = top_5.union(bottom_5)

candle_data = []

for county in selected_counties:
    county_df = df[df['County'] == county].sort_values('Date')
    values = county_df['Daily Max 8-hour Ozone Concentration']
    
    open_val = values.iloc[0] 
    high_val = values.max()
    low_val = values.min()
    
    candle_data.append({
        'County': county,
        'Open': open_val,
        'High': high_val,
        'Low': low_val,
        'Close': open_val,  
        'Color': 'red' if county in top_5 else 'green'
    })

candle_df = pd.DataFrame(candle_data)

fig = go.Figure()

for _, row in candle_df.iterrows():
    fig.add_trace(go.Candlestick(
        x = [row['County']],
        open = [row['Open']],
        high = [row['High']],
        low = [row['Low']],
        close = [row['Open']],
        increasing_line_color = row['Color'],
        decreasing_line_color = row['Color'],
        showlegend = False
    ))

fig.update_layout(
    title = 'Ozone Concentration Range (Top 5 in Red, Bottom 5 in Green)',
    yaxis_title = 'Ozone Concentration (ppm)',
    xaxis_title = 'County',
    xaxis_type = 'category',
    xaxis_tickangle = -45
)

fig.show()

county_avg = df.groupby(['County', 'Site Latitude', 'Site Longitude'])[
    'Daily Max 8-hour Ozone Concentration'
].mean().reset_index()
county_avg.sort_values(by = 'Daily Max 8-hour Ozone Concentration', ascending = True, inplace = True)
fig = px.scatter_geo(
    county_avg,
    lat = 'Site Latitude',
    lon = 'Site Longitude',
    color = 'Daily Max 8-hour Ozone Concentration',
    hover_name = 'County',
    color_continuous_scale = 'YlOrRd',
    scope = 'usa',
    title = 'Average Ozone Concentration by County (Lowest to Highest)'
)

fig.update_layout(margin={"r": 0,"t": 40,"l": 0,"b": 0})
fig.show()

# Grouping data for each visualization in order to show the difference
df_method = df.copy()
df_method['Date'] = df_method['Date'].astype(str)
method_avg = df_method.groupby(['Local Site Name', 'Method Code']).agg({
    'Daily AQI Value': 'mean',
    'Date': 'count'
}).reset_index()
method_avg.sort_values(by = ['Daily AQI Value', 'Method Code'], ascending = [False, True])
method_avg = method_avg[~method_avg.isin(['unknown']).any(axis = 1)]
method_avg['Repeated Values'] = method_avg['Local Site Name'].map(method_avg['Local Site Name'].value_counts())
method_avg = method_avg[(method_avg['Repeated Values'] > 1)]
method_avg = method_avg.rename(columns = {'Date' : 'Amount Of Days Used', 'Daily AQI Value': 'Daily AQI Value Average'})
display(method_avg)
method_analysis = method_avg.merge(
    df_method,
    on = 'Local Site Name',
    how = 'inner',
    suffixes=('', '_drop')
)
print(f"There are only {method_analysis['Local Site Name'].unique()} that have used more than one known method for gaining information on AQI and Ozone. We will use these sites to show clear difference.")

method_analysis = method_analysis.rename(columns={'Local Site Name': 'Site'})
method_analysis['Date'] = pd.to_datetime(df['Date'])
method_analysis = method_analysis.sort_values(by = 'Date', ascending = True)
method_analysis['Month'] = method_analysis['Date'].dt.to_period('M').dt.to_timestamp()
daily_avg = method_analysis.groupby(['Month', 'Site', 'Method Code'])['Daily AQI Value'].mean().reset_index()

fig = px.line(
    daily_avg,
    x = 'Month',
    y = 'Daily AQI Value',
    color = 'Method Code',
    facet_row = 'Site',
    title = 'AQI Value Over Time by Method Code',
    markers = True
)

fig.update_layout(
    height = 900,
    yaxis_title = 'Daily AQI Value',
    xaxis_title = 'Month',
    legend_title = 'Method Code'
)

fig.show()

new_df = df.copy()
new_df['County'] = new_df['County Group'].str.replace(r'\s*\(Top 5\)|\s*\(Bottom 5\)', '', regex = True)
weekend_df = new_df[new_df['Type Of Day'] == 'Weekend']
weekday_df = new_df[new_df['Type Of Day'] == 'Weekday']
county_avg_weekend = weekend_df.groupby('County')['Daily Max 8-hour Ozone Concentration'].mean()
county_avg_weekday = weekday_df.groupby('County')['Daily Max 8-hour Ozone Concentration'].mean()
county_avg_weekend_df = county_avg_weekend.reset_index().rename(columns={'Daily Max 8-hour Ozone Concentration': 'Weekend Avg'})
county_avg_weekday_df = county_avg_weekday.reset_index().rename(columns={'Daily Max 8-hour Ozone Concentration': 'Weekday Avg'})
urban_activity = county_avg_weekend_df.merge(county_avg_weekday_df, on = 'County', how = 'inner')

fig = px.bar(
    urban_activity,
    x = 'County',
    y = ['Weekend Avg', 'Weekday Avg'],
    barmode = 'group',
    title = 'Average Ozone Concentration: Weekend vs Weekday by County',
    labels = {'value': 'Ozone Concentration (ppm)', 'variable': 'Day Type'}
)

fig.update_layout(
    xaxis_title='County',
    yaxis_title='Ozone Concentration (ppm)',
    xaxis_tickangle = 45,
    width = 1000, height = 600
)

fig.show()

# bonus plot

dataset = df.copy()
avg_point = dataset.groupby(['Site Latitude', 'Site Longitude']).agg({
    'Daily Max 8-hour Ozone Concentration' : 'mean'
}).reset_index()
avg_point = avg_point.rename(columns = {'Daily Max 8-hour Ozone Concentration' : 'Ozone Concentration'})
fig = px.density_mapbox(avg_point, lat = 'Site Latitude', lon = 'Site Longitude', z = 'Ozone Concentration',
                        title = 'Bonus Plot: Heatmap For Daily Max Ozone Concentration (Average)',
                        radius = 8,
                        center = dict(lat = 36, lon = -119),
                        zoom = 3,
                        mapbox_style = 'open-street-map',
                        color_continuous_scale = 'Plasma')
fig.show()