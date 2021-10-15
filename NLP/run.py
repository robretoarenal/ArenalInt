import streamlit as st
import pandas as pd
from utils import PredictSentiment,SentimentTrain,GetData

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.title('NLP - Sentiment Analysis')
    #st.write('This model tells whether a sentence is positive or negative.')
    st.markdown("""
        This application contains a trained sentiment analysis model, capable of telling 
        whether a sentence is positive or negative. The data used to develop this model comes from 3 different
        sources, all of them being customers' opinion about the service/product they consumed.""")

    #st.sidebar.header('Header')
    #st.sidebar.write('The preprocessing step was performed using lemmatization as well as removing stop words and punctuation.')

    st.write('Take a look at the different datasets:')
    dataset=st.radio(label='',options = ('None','Yelp','IMDB','Amazon'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    #Create an instance of GetData class
    gd = GetData('SentimentAnalysis/Data')
    
    col1, col2 = st.columns(2)

    if dataset=='Yelp':
        df_pos,df_neg=gd.dataLoad(1)
        col1.header("Positive")
        col1.dataframe(pd.DataFrame(df_pos).reset_index(drop=True),width=450,height=450)
        #col1.table(pd.DataFrame(df_pos).reset_index(drop=True))
        col2.header("Negative")
        col2.dataframe(pd.DataFrame(df_neg).reset_index(drop=True),width=450,height=450)
        #col2.table(pd.DataFrame(df_neg).reset_index(drop=True))
    elif dataset=='IMDB':
        df_pos,df_neg=gd.dataLoad(2)
        col1.header("Positive")
        col1.dataframe(pd.DataFrame(df_pos).reset_index(drop=True),width=450,height=450)
        col2.header("Negative")
        col2.dataframe(pd.DataFrame(df_neg).reset_index(drop=True),width=450,height=450)
    elif dataset=='Amazon':
        df_pos,df_neg=gd.dataLoad(3)
        col1.header("Positive")
        col1.dataframe(pd.DataFrame(df_pos).reset_index(drop=True),width=450,height=450)
        col2.header("Negative")
        col2.dataframe(pd.DataFrame(df_neg).reset_index(drop=True),width=450,height=450)
    
    user_input = st.text_input("Write your sentence: ", ' ')
    inp_arr = [user_input]
    if st.button('Predict'):
        pred,prob,df_tokens=PredictSentiment().predict(inp_arr)
        st.header(user_input)
        if pred[0]==0:
            string = "{}% Negative!:thumbsdown:".format(round(prob,2))
            cmap='OrRd'
            df_tokens.Coef = df_tokens.Coef * -1
            #high=-10
            st.header(string)
        else: 
            string = "{}% Positive!:thumbsup:".format(round(prob,2))
            cmap='Blues'
            #high=10
            st.header(string)

        #df.set_index('TOKEN', inplace=True)
        #st.dataframe(df.assign(hack='').set_index('hack'))
        #pd.set_option('max_colwidth',600)

        #df.style.set_properties(subset=['Coef'],**{'width':'600px'})
        #st.dataframe(df.style.background_gradient(cmap=cmap).set_properties())

        #st.table(df.assign(hack='').set_index('hack'))
        st.table(df_tokens.set_index('TOKEN').sort_values(by='Coef',ascending=False).style.background_gradient(cmap=cmap))
