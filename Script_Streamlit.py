import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime



from streamlit_extras.let_it_rain import rain


st.set_page_config(
    page_title="Bike Problem Analysis",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


# Load the data
data = pd.read_csv('bike-sharing-hourly.csv')

#
 
def bike_problems_page(data):
    st.title("Bike Sharing Dashboard")
    
    with st.expander("Our Goal and Project Overview", expanded=False):
        st.markdown("""
            ## Our client's goal is to refine their bike-sharing service, making it more efficient and aligned with user needs. 
            
            This project is designed around two pivotal elements: conducting an in-depth analysis of how users engage with the service and developing a predictive model to estimate hourly bike usageThe analysis will identify usage patterns, peak demand periods, and areas for service improvement, aiming to enhance customer satisfaction and operational efficiency. The predictive model, on the other hand, will enable precise bike allocation, reducing unnecessary expenses and improving service responsiveness.
            
            Together, these efforts will provide a strategic roadmap for optimizing the bike-sharing service, ensuring it meets the dynamic demands of urban mobility and sets a benchmark for future innovation.
        """)
        main_dashboard_image = 'https://www.gannett-cdn.com/-mm-/b675ed74abcc46f3c7f21d67c2dda87f3e2b4f10/c=4-0-2048-1536/local/-/media/USATODAY/USATODAY/2013/09/30/1380563336010-11-Washington-Flickr-Mr-T-in-DC.jpg?width=960'
        st.image(main_dashboard_image, caption="Bike Sharing in Action")

    # Static metrics
    st.header("Key Metrics and Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Bike Rentals', "3,292,679")
    with col2:
        st.metric('Registered Users', "2,672,662")
    with col3:
        st.metric('Casual Users', "620,017")

    # Data Visualization (Example with Altair)
    st.header("Bike Rentals by Hour")
    chart_data = data.groupby('hr')['cnt'].sum().reset_index()
    c = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('hr:O', axis = alt.Axis(title='Hour of the Day')),
        y=alt.Y('cnt:Q', axis = alt.Axis(title='Total Bike Rentals')),
        tooltip=['hr', 'cnt']
    ).properties(width=700, height=400)
    st.altair_chart(c, use_container_width=True)
    

    # Dataset Overview
    st.header("Dataset Overview")
    st.dataframe(data)  # Display filtered data
    


def profiles_page():
    st.title('User Profiles')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Total Rental Count by Year')
        yearly_comparison = data.groupby('yr')['cnt'].sum().reset_index()
        fig = px.pie(yearly_comparison, values='cnt', names='yr', 
                     title='Total Rental Count by Year')
        st.plotly_chart(fig)
        
        st.subheader('Average Rental Count on Holidays vs. Non-Holidays')
        holiday_impact = data.groupby('holiday')['cnt'].mean().reset_index()
        fig = px.bar(holiday_impact, x='holiday', y='cnt', 
                     labels={'holiday': 'Holiday', 'cnt': 'Average Rental Count'},
                     title='Average Rental Count on Holidays vs. Non-Holidays')
        st.plotly_chart(fig)
        
        hourly_usage_weekday = data.groupby(['weekday', 'hr'])['cnt'].median().reset_index()
        weekday_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        hourly_usage_weekday['weekday'] = hourly_usage_weekday['weekday'].map(weekday_names)
        fig = px.line(hourly_usage_weekday, x='hr', y='cnt', color='weekday',
                      labels={'hr': 'Hour of the Day', 'cnt': 'Median Rental Count'},
                      title='Hourly Rental Distribution by Day of the Week')
        fig.update_xaxes(tick0=0, dtick=1)
        st.plotly_chart(fig)

    with col2: 

        st.subheader('Median Registered vs. Casual User Counts by Hour')
        hourly_registered_casual = data.groupby('hr')[['registered', 'casual']].median().reset_index()
        fig = px.line(hourly_registered_casual, x='hr', y=['registered', 'casual'], 
                      labels={'hr': 'Hour of the Day', 'value': 'Count', 'variable': 'User Type'},
                      title='Median Registered vs. Casual User Counts by Hour')
        fig.update_xaxes(tick0=0, dtick=1)
        st.plotly_chart(fig)
        
        st.subheader('Registered vs. Casual User Counts Over Time')
        registered_casual = data.groupby('dteday')[['registered', 'casual']].sum().reset_index()
        fig = px.line(registered_casual, x='dteday', y=['registered', 'casual'], 
                      labels={'dteday': 'Date', 'value': 'Count', 'variable': 'User Type'},
                      title='Registered vs. Casual User Counts Over Time')
        st.plotly_chart(fig)
        data['temp_combined'] = (data['temp'] + data['atemp']) / 2
        data['hum_binned'] = pd.cut(data['hum'], bins=[0, 0.33, 0.66, 1], labels=['Low Humidity', 'Medium Humidity', 'High Humidity'], include_lowest=True)
        data['temp_binned'] = pd.cut(data['temp_combined'], bins=[0, 0.33, 0.66, 1], labels=['Low Temp', 'Medium Temp', 'High Temp'], include_lowest=True)
        data['windspeed_binned'] = pd.cut(data['windspeed'], bins=[0, 0.33, 0.66, 1], labels=['Low Wind', 'Medium Wind', 'High Wind'], include_lowest=True)
        grouped_windspeed = data.groupby('windspeed_binned')['cnt'].sum().reset_index()
        grouped_humidity = data.groupby('hum_binned')['cnt'].sum().reset_index()
        grouped_temp = data.groupby('temp_binned')['cnt'].sum().reset_index()
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Count by Windspeed', 'Count by Humidity', 'Count by Temperature'))

        # Plot the windspeed binned
        fig.add_trace(go.Bar(x=grouped_windspeed['windspeed_binned'], y=grouped_windspeed['cnt'],
                            name='Windspeed', marker_color='blue'), row=1, col=1)

        # Plot the humidity binned
        fig.add_trace(go.Bar(x=grouped_humidity['hum_binned'], y=grouped_humidity['cnt'],
                            name='Humidity', marker_color='orange'), row=1, col=2)

        # Plot the temperature binned
        fig.add_trace(go.Bar(x=grouped_temp['temp_binned'], y=grouped_temp['cnt'],
                            name='Temperature', marker_color='green'), row=1, col=3)

        # Update layout
        fig.update_layout(title='Rental Counts by Weather Factors',
                        showlegend=False,
                        height=500,
                        width=900)

        # Update title text size to fit above the graphs
        fig.update_layout(title_text='Rental Counts by Weather Factors', title_x=0.5)

        # Show the plot in Streamlit
        st.plotly_chart(fig)


    
        
        
# Load the data
data_pre = pd.read_csv('bike_sharing_modified.csv')

def get_season(mt):
        if mt in [12, 1, 2]:
            return 1
        elif mt in [3, 4, 5]:
            return 2
        elif mt in [6, 7, 8]:
            return 3
        else:
            return 4

def predictor_page():
    st.title('Predictor')       
    #from Prediction.py import model_prediction

    with st.form(key='input_form'):
        humidity_feeling = st.selectbox('Select weather condition', ['High Humidity', 'Medium Humidity', 'Low Humidity'])
        wind_feeling = st.selectbox('Select wind condition', ['High Wind', 'Medium Wind', 'Low Wind'])
        temp_feeling = st.selectbox('Select temperature condition', ['High Temperature', 'Medium Temperature', 'Low Temperature'])
        temp_actual = st.number_input("Enter the actual temperature in Celsius")
        # Time_of_the_day = st.selectbox('Select time of the day', ['Early Morning', 'Morning', 'Afternoon', 'Evening'])
        # season = st.selectbox('Select season', [1, 2, 3, 4])
        # year = st.selectbox('Select year', [0, 1])
        # month = st.selectbox('Select month', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) 
        date_pred = st.date_input("Select the date", format="MM.DD.YYYY")
        hour = st.selectbox('Select hour', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])  
        # hr_time = st.time_input("Select the time")
        # weekday = st.selectbox('Select weekday', [0, 1, 2, 3, 4, 5, 6])       
        # rush_hour = st.selectbox('Select rush hour', [0, 1])    
        # humidity = st.number_input("Enter the humidity percentage")
        # wind_speed = st.number_input("Enter the wind speed in km/h")
        # temperature = st.number_input("Enter the temperature normalised")
        # temperature_difference_perceived = st.number_input("Enter the temperature difference perceived")
        # temperature_previous_hour = st.number_input("Enter the temperature of the previous hour")
        # humidity_previous_hour = st.number_input("Enter the humidity of the previous hour")
        daily_rentals = st.number_input("Enter the bike rentals from the last 24 hours")
        discomfort_index = st.number_input("Enter the discomfort index")
        submit_button = st.form_submit_button(label='Predict')
    
    hour_bins = [0, 6, 12, 18, 24]
    bin_labels = ['Early Morning', 'Morning', 'Afternoon', 'Evening']
    # Create the new column
    data['hour_bin'] = pd.cut(data['hr'], bins=hour_bins, labels=bin_labels, right=False)

    if submit_button:
        # Data preprocessing
        input_data = pd.DataFrame({
            'hum_binned_high': [ 1 if humidity_feeling== 'High Humidity' else 0],
            'hum_binned_med': [1 if humidity_feeling== 'Medium Humidity' else 0],
            'hum_binned_low': [1 if humidity_feeling== 'Low Humidity' else 0],
            'hour_bin_early': [1 if (hour > 0) and (hour<=6) else 0],
            'hour_bin_morning': [1 if (hour > 6) and (hour<=12) else 0],
            'hour_bin_afternoon': [1 if (hour > 12) and (hour<=18) else 0],
            'hour_bin_evening': [1 if (hour > 18) and (hour<=0) else 0],
            'season': [get_season(date_pred.month)],
            'year': [1],
            'month': [date_pred.month],
            'hour': [hour],
            'weekday': [date_pred.weekday()],
            'rush_hour': [1 if (hour==8) or (hour >= 16 and hour <= 18) else 0],
            'hum': [-0.55 if humidity_feeling== 'High Humidity' else 0.53 if humidity_feeling== 'Low Humidity' else 1.93],
            'windspeed': [-0.45 if wind_feeling== 'High Wind' else 0.27 if wind_feeling== 'Low Wind' else 2.71],
            'temp_combined': [temp_actual/41],
            'temp_diff_perceived': [abs(temp_actual/41 - 2.09) if temp_feeling== 'High Temperature' else abs(temp_actual/41 - 0.41 ) if temp_feeling== 'Medium Temperature' else abs(temp_actual/41+0.32)],
            'temp_prev': [temp_actual -  np.random.rand()],
            'hum_prev': [-0.55-np.random.rand() if humidity_feeling== 'High Humidity' else 0.53-np.random.rand() if humidity_feeling== 'Low Humidity' else 1.93-np.random.rand()],
            'daily_rentals': [daily_rentals],
            'Weather Discomfort Index': [np.random.rand()]
            })
        
        from Prediction import model_prediction

        prediction = model_prediction(input_data)

        st.title(f"Predicted bike rentals: {round(prediction)}")
            

        
def prediction_model_page():
    st.title("How Do We Predict?")
    st.subheader('Exploratory Data Analysis')
    # Dataset Overview
    st.header(" New Dataset Overview")
    st.dataframe(data_pre)  # Display filtered data

    col1, col2 = st.columns([1, 2])  # Adjust column ratios if necessary
    
    with col1:

     st.write("Features and Their Types:")
     features_types = pd.DataFrame(data_pre.dtypes, columns=['Type']).reset_index()
     features_types.columns = ['Feature', 'Type']  # Renaming the columns for clarity
     st.dataframe(features_types)

     st.title('Model Used: LightGBM Regressor')
     st.title('R^2 Score: 0.93')
     st.subheader('Exploratory Data Analysis')
  
    
      
    with col2:
     st.subheader('Dataset Feature Descriptions')
     st.markdown("""

     - **dteday**: The date on which bike rentals were recorded.
     - **season**: Categorical feature indicating the season (1: winter, 2: spring, 3: summer, 4: fall).
     - **yr**: Year (0: 2011, 1: 2012).
     - **mnth**: Month of the year (1 to 12).
     - **hr**: Hour of the day (0 to 23).
     - **holiday**: Whether the day is a holiday (0: No, 1: Yes).
     - **weekday**: Day of the week (0: Sunday, 1: Monday, ..., 6: Saturday).
     - **workingday**: Whether the day is a working day (0: No, 1: Yes).
     - **weathersit**: Categorical variable indicating the weather situation (e.g., 1: Clear, 2: Mist, 3: Light Snow, 4: Heavy Rain).
     - **temp**: Normalized temperature in Celsius. The values are divided by (max temp - min temp), (0 to 1).
     - **atemp**: Normalized feeling temperature in Celsius.
     - **hum**: Normalized humidity. The values are divided by 100 (max).
     - **windspeed**: Normalized wind speed. The values are divided by the max value.
     - Additional features such as **rush_hour**, **temp_diff_perceived**, **weather_intensity**, **temp_prev**, **hum_prev**, **windspeed_prev**, **weather_change_indicator**, **temp_median**, **daily_rentals**, and **Weather Discomfort Index** seem to be derived or calculated based on the primary features or external data to potentially enhance the analysis. These include:
     - **rush_hour**: Indicates if the time is within rush hours (likely a binary indicator).
     - **temp_diff_perceived**: A calculated difference between perceived temperature and actual temperature.
     - **weather_intensity**: A metric indicating the intensity of the weather condition.
     - **temp_prev**, **hum_prev**, **windspeed_prev**: The previous measurements of temperature, humidity, and wind speed, respectively.
     - **weather_change_indicator**: Indicates changes in weather conditions compared to previous measurements.
     - **temp_median**: The median temperature for some period.
     - **daily_rentals**: The total number of bike rentals for the day.
     - **Weather Discomfort Index**: A calculated index indicating the level of discomfort due to weather conditions """, unsafe_allow_html=True)
   
     









def rating_system(rec_id):
    rating = st.slider(f"Rate Recommendation {rec_id}", 0, 5, key=f'slider_{rec_id}')
    
    if rating in [1, 2, 3]:
        # Trigger the thumbs down emoji rain animation if rating is 1, 2, or 3
        rain(
            emoji="👎",
            font_size=90,
            falling_speed=1.5,
            animation_length="0.1s",  # Adjust this as needed based on your rain function's implementation
        )
    else:
        # If the rating is 4 or 5, trigger balloons
        st.balloons()


def create_recommendation(rec_id, title, content, image_url):
    st.markdown(f"### {title}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(content)
        rating_system(rec_id)
    with col2:
        st.image(image_url)

def predictions_recommendations_page():
    st.title("Predictions / Recommendations")

    # Create multiple recommendations
    create_recommendation(
        rec_id=1,
        title="Promotional Pricing in Off-Peak Hours",
        content="Analysis of rental trends indicates a lower tendency for bike rentals during certain hours of the day. To counterbalance this, implementing promotional pricing during these off-peak hours could stimulate demand. For instance, offering reduced rates in the early morning or late evening encourages people to use bike sharing for their commute or leisure activities, leading to a more consistent rental pattern throughout the day",
        image_url="https://shifttransit.net/wp-content/uploads/2020/11/Ad-Poster-final-creative-2020-1024x736.png"
    )
    create_recommendation(
        rec_id=2,
        title="Incentive Campaign: 15 Days of Free Use for the First Hour",
        content="Introducing an incentive campaign, such as providing 15 days of free use for the first hour, can attract new users and encourage regular customers to increase their usage. This strategy not only boosts rentals during the promotional period but also has the potential to convert temporary users into regular ones, as they experience the convenience and benefits of bike sharing firsthand",
        image_url="https://cdn01.bcycle.com/libraries/images/librariesprovider74/Home-Page/april-start-bike-jackson-hole.jpg?sfvrsn=447421c5_0"
    )
    
    create_recommendation(
        rec_id=3,
        title="Referral Discount Program",
        content="A referral discount program can leverage existing users to attract new customers. By offering discounts to both the referrer and the referee, the service incentivizes current users to promote bike sharing within their social circles. This word-of-mouth marketing approach is cost-effective and can significantly increase the user base, resulting in higher rental volumes",
        image_url="https://www.mobibikes.ca/sites/default/files/ressources/18.06.28_bestie_2_1_0.png"
    )
    create_recommendation(
        rec_id=4,
        title="Seasonal and Event Promotions",
        content="Capitalizing on seasonal changes and local events by offering tailored promotions can significantly boost rentals. During warmer months or in conjunction with local festivals, parades, or sporting events, bike sharing can become the preferred mode of transportation. Offering event-specific discounts or creating themed rides that explore seasonal attractions in Washington can attract both tourists and locals, encouraging more frequent rentals.",
        image_url="https://scontent.fmad22-1.fna.fbcdn.net/v/t39.30808-6/321550532_894378871591363_7504091129692758603_n.png?_nc_cat=106&ccb=1-7&_nc_sid=5f2048&_nc_ohc=mu5DZpm-4rsAX-V0aVV&_nc_ht=scontent.fmad22-1.fna&oh=00_AfDshi_2_kk9X8EG70z1mxGNtS87ATnmSUqJR6JqsVP8bg&oe=65F9D9D7"
    )

    st.markdown("## Conclusion")
    st.write("The success of bike sharing services in Washington hinges on the ability to adapt to and influence user behavior. By implementing targeted strategies such as promotional pricing during off-peak hours, incentive campaigns for new users, referral discount programs, and seasonal promotions, bike sharing services can significantly increase their hourly rentals.")
    st.write("These recommendations not only aim to elevate the usage rates but also enhance the overall visibility and appeal of bike sharing as a sustainable transport option in Washington. Through innovative marketing and customer engagement, bike sharing services can achieve sustained growth and contribute to a greener, more mobile urban environment")
    st.markdown("### Your Recommendations")
    user_input = st.text_area("Share your thoughts", "")
    if st.button("Submit"):
        st.write("Thank you for your recommendation!")

def detect_outliers(series, threshold=3):
    mean = series.mean()
    std = series.std()
    outliers = series[(series - mean).abs() > threshold * std]
    return outliers

# Make sure to pass the data when you call the function in the app
# data should be defined outside of this function, as a global variable or loaded elsewhere
# if __name__ == "__main__":
#     technical_annex_page(data)

    st.write("Content for Technical Annex...")

# Dictionary of pages
pages = {
    "Bike Problems": bike_problems_page,
    "Profiles": profiles_page,
    "Predictor": predictor_page,
    "How Do We Predict?": prediction_model_page,
    "Predictions / Recommendations": predictions_recommendations_page,
}

def main():
    st.sidebar.title("🚴 Bike Sharing Dashboard!")
    page = st.sidebar.selectbox("Our Road Insights 🛣", ["Bike Problems", "Profiles", "Predictor", "How Do We Predict?", "Predictions / Recommendations"])
    if page == "Bike Problems":
        bike_problems_page(data)
    elif page == "Profiles":
        profiles_page()
    elif page == "Predictor":
        predictor_page()
    elif page == "How Do We Predict?":
        prediction_model_page()
    elif page == "Predictions / Recommendations":
        predictions_recommendations_page()


# Run the app
data = pd.read_csv('bike-sharing-hourly.csv')
if __name__ == "__main__":
    main()
  