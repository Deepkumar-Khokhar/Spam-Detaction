import streamlit as st
import pickle
import re
import string
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def tranceform_text(text):
    text = text.lower()
    text = re.split(r'[^\w]+',text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in STOPWORDS and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load the model
model = pickle.load(open('mnb.pkl', 'rb'))
tfidf = pickle.load(open('vectorize.pkl', 'rb'))

def main():
    st.title("Spam Classifier")
    st.write("Welcome to the Spam Detection App! Enter a message below and we'll tell you if it's spam or not.")
    
    message = st.text_area("Enter a message:")
    


    
    if st.button("Check"):
        tranceformed_sms = tranceform_text(message)
        vector_input = tfidf.transform([tranceformed_sms])
        result = model.predict(vector_input)

        if result == 1:
            st.header("This message is spam!")
        else:
            st.header("This message is not spam.")
        
if __name__ == "__main__":
    main()
