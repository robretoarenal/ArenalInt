import streamlit as st
from utils import PredictSentiment

if __name__ == "__main__":

    st.title('NLP - Sentiment Analysis')
    st.write('This model tells whether a sentence is positive or negative.')

    #image = Image.open('nfl-1.jpg')
    #st.image(image, use_column_width=True)
    user_input = st.text_input("Write your sentence: ", ' ')
    inp_arr = [user_input]
    if st.button('Predict'):
        pred=PredictSentiment().predict(inp_arr)
        st.header(user_input)
        st.header(pred)
        #st.table(df.assign(hack='').set_index('hack'))
