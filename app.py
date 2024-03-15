# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components
# import streamlit_analytics
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download('wordnet')
from sentence_transformers import SentenceTransformer
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import time

st.set_page_config(page_title="Mental disorder by description", page_icon="🤖")

def convert_string_to_numpy_array(s):
    '''Function to convert a string to a NumPy array'''
    numbers_list = re.findall(r'-?\d+\.\d+', s)
    return np.array(numbers_list, dtype=np.float64)

#load the model
@st.cache_resource
def get_models():
  st.write('*Loading the model...*')
  name = "stsb-bert-large"
  model = SentenceTransformer(name)
  st.write("*The app is loaded and ready to use! :tada:*")
  lemmatizer = WordNetLemmatizer()
  return model, lemmatizer

model, lemmatizer = get_models()
stop_words = set(stopwords.words('english'))

#load the dataframe with disorder embeddings
@st.cache_data  # 👈 Add the caching decorator
def load_data():
    df_icd = pd.read_csv('icd_embedded.csv')
    df_icd['numpy_array'] = df_icd['Embeddings'].apply(convert_string_to_numpy_array)
    icd_embeddings = np.array(df_icd["numpy_array"].tolist())
    return df_icd, icd_embeddings

df_icd, icd_embeddings = load_data()

#create a list of disease names
@st.cache_data  # 👈 Add the caching decorator
def create_disease_list():
    disease_names = []
    for name in df_icd["Disease"]:
        disease_names.append(name)
    return disease_names

disease_names = create_disease_list()

if 'descriptions' not in st.session_state:
  st.session_state.descriptions = []

def similarity_top(descr_emb, disorder_embs):
  # reshaping the character_embedding to match the shape of mental_disorder_embeddings
  descr_emb = descr_emb.reshape(1, -1)
  # calculating the cosine similarity
  similarity_scores = cosine_similarity(disorder_embs, descr_emb)

  scores_names = []
  for score, name in zip(similarity_scores, disease_names):
      data = {"disease_name": name, "similarity_score": score}
      scores_names.append(data)

  scores_names = sorted(scores_names, key=lambda x: x['similarity_score'], reverse=True)

  results = []

  for item in scores_names:
    disease_name = item['disease_name']
    similarity_score = item['similarity_score'][0]
    results.append((disease_name, similarity_score))

  return results[:5]

def vis_results_2d(input_embed):

    # performing dimensionality reduction using PCA
    pca = PCA(n_components=2)
    disease_embeddings_2d = pca.fit_transform(icd_embeddings)

    # creating a DataFrame for disease embeddings plot
    disease_data_df = pd.DataFrame(disease_embeddings_2d, columns=['PC1', 'PC2'])
    disease_data_df['Type'] = 'Disease'
    disease_data_df['Name'] = disease_names

    input_embed_2d = input_embed.reshape(1, -1)
    input_embed_2d = pca.transform(input_embed_2d)

    # creating a DataFrame for character embedding plot
    pca_2d = pd.DataFrame(input_embed_2d, columns=['PC1', 'PC2'])
    pca_2d['Type'] = 'Character'
    pca_2d['Your character'] = 'Your character'

    # concatenating the two DataFrames
    combined_2d = pd.concat([disease_data_df, pca_2d], ignore_index=True)

    # creating an interactive 3D scatter plot
    fig = px.scatter(combined_2d, x='PC1', y='PC2', text='Name', color='Type', symbol='Type', width=800, height=800)
    fig.show()

    
def vis_results_3d(input_embed):

    # performing dimensionality reduction using PCA
    pca = PCA(n_components=3)
    disease_embeddings_3d = pca.fit_transform(icd_embeddings)

    # creating a DataFrame for disease embeddings plot
    disease_data_df = pd.DataFrame(disease_embeddings_3d, columns=['PC1', 'PC2', 'PC3'])
    disease_data_df['Type'] = 'Disease'
    disease_data_df['Name'] = disease_names

    input_embed_2d = input_embed.reshape(1, -1)
    input_embed_3d = pca.transform(input_embed_2d)
    
    # creating a DataFrame for character embedding plot
    pca_3d = pd.DataFrame(input_embed_3d, columns=['PC1', 'PC2', 'PC3'])
    pca_3d['Type'] = 'Character'
    pca_3d['Your character'] = 'Your character'

    # concatenating the two DataFrames
    combined_3d = pd.concat([disease_data_df, pca_3d], ignore_index=True)

    # creating an interactive 3D scatter plot
    fig = px.scatter_3d(combined_3d, x='PC1', y='PC2', z='PC3', text='Name', color='Type', symbol='Type', width=800, height=800)
    fig.show()

# Configure Streamlit page and state
st.title("Detect your character's mental disorder! :books: :mag:")
st.markdown(
    "This mini-app predicts top-5 most likely mental disorders based on your description. The more information you provide, the more informative the results will be."
)
st.caption("NOTE: This is just a preliminary diagnosis based on previous medical data by ICD\n Consult a doctor for better analysis.")
input = st.text_input(label="Your description", placeholder="Insert a description of your character")
if input:
    input_embed = model.encode(input)
    sim_score = similarity_top(input_embed, icd_embeddings)
    i = 1
    nums = {1: 'one', 2: 'two', 3: 'three', 4:'four', 5:'five'}
    for dis, value in sim_score:
        st.write(f":green[*Prediction number*] :{i}: :")
        st.write(f"{dis} (similarity score:", value, ")")
        i+= 1
        
    text_spinner_placeholder = st.empty()
    # with st.spinner("Please wait while your visualizations are being generated..."):
    #     time.sleep(5)
    # vis_results_2d(input_embed)
    # vis_results_3d(input_embed)
if st.button("Book An Appoitment")
redirect_js = """ <script>window.location.href='localhost:3000/appointment.html'</script>"""
st.write(redirect_js, unsafe_allow_html=True)
    

                    
