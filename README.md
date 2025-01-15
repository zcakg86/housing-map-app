# Housing Map App
This personal project is a work in progress. It implements an interactive web map, utilizing spatial data and AI to assist in housing search and housing market knowledge regarding location.
Listings data is combined with historic sales data (for trends), and points of interest (in this example daycares: targeting those interested in accessibility to childcare).

URL when running: https://turbo-cod-r4w75j5jjrqhx64p-8501.app.github.dev/ 

Streamlit is used to build the app. The project is hosted with Codespaces. Code is all written in Python. OpenAI's GPT-3.5-turbo-0125 model is used for the chat function.

![screenshot of the app](streamlit_app.png "App screenshot")
## Aims
The primary aim is to provide develop Python programming skills, utilizing cloud platforms and spatial tools to develop ways to present housing market analysis for interested individual users.
* Look into spatial representations: I want to try out h3index as a hierarchical grid spatial representation rather than an xy-coordinate representation, and how this might assist with implementing machine learning.
* Combine data on property listings, points of interest, and transportation to assist with location choice.
* Implement ML with neural networks to identify trends in data. 
* Explore retrieval augmented generation with foundation models to interact user queries with presented data.
  
## Next Steps
* Program and visualize more complex temporal and spatial analysis utilizing historical sales data. Use of machine learning methods rather than traditional statistical/econometric methods.
* Improve visualization of relative prices within the map - bring in attention scores from neural network. 

Nice to Haves:
* Allow AI chat response to interact with the map.
* Bring in transport network data for accessibility considerations.
* Call listings and POI APIs directly within the app (currently the data app uses static data stored as CSV files, and listings are out of date).
* Build on the functionality of the AI chat. Explore using additional AI tools and contextual data to assist the RAG agent.

## Data
Residential Sales data: kingco_sales.csv\
https://www.kaggle.com/datasets/andykrause/kingcountysales/data \
Andy Krause\
Data produced from data provided by the King County Department of Assessments.

Residential Listings: Web scraped Zillow listing data from RapidAPI sources\
https://rapidapi.com/apimaker/api/zillow-com1 \
https://rapidapi.com/s.mahmoud97/api/zillow56

Point of interest data:\
Foursquare API

Icon: from Noun Project\
Firza Alamsyah

### To be used in further analysis:
Income by Census Tract, 2000, 2006-2022 (American Community Survey 5-year period) American Community Survey.
(Income data planned to be combined with changes in house prices for analysis, rather than to compare residents' incomes by neighbourhood).
Steven Manson, Jonathan Schroeder, David Van Riper, Katherine Knowles, Tracy Kugler, Finn Roberts, and Steven Ruggles. IPUMS National Historical Geographic Information System: Version 18.0 [dataset]. Minneapolis, MN: IPUMS. 2023. http://doi.org/10.18128/D050.V18.0
https://www.nhgis.org/

King County Transit lines
King County Transit stops points
https://gis-kingcounty.opendata.arcgis.com/

30-Year Fixed Rate Mortgage Average in the United States
https://fred.stlouisfed.org/series/MORTGAGE30US
