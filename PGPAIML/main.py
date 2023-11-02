import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from streamlit_modal import Modal
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px  
import os
import openai
import json
import textwrap

class Config:
  
   deployment_name =  "gpt-3.5"
   model_name =  "gpt-3.5-turbo"
  
os.environ["OPENAI_API_KEY"]= "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"

openai.api_type = "open_ai"
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title = 'GenAI', layout="wide")

##################### All Functions ####################

def line_break(comp, num_breaks):
    for i in range(num_breaks):
        comp.markdown('\n')
        
def custom_info(message):
    st.sidebar.markdown(f'<div class="custom-st-info">{message}</div>', unsafe_allow_html=True)
            
global_prompt = (
    "Please answer the following questions about a company. If you don't know the answer, you will be provided with text output from google search snippet to support your response; you may choose to augment your answer from there. First try yourself, then take support of text.'\n"
    "- The full address of the headquarters\n"
    "- Every phone number listed\n"
    "- All email addresses provided\n"
    "- Detailed information about the industry\n"
    "- Names and details of all subsidiary companies or brands\n"
    "- The closest SIC code for the company's industry.\n"
)
rewrite_prompt = (
    "Please check the following answer. Correct them as per your best knowledge and add them if they are not available.'\n"
    "1. The full address of the headquarters\n"
    "2. Every phone number listed\n"
    "3. All email addresses provided\n"
    "4. Detailed information about the industry\n"
    "5. Names and details of all subsidiary companies or brands\n"
    "6. The closest SIC code for the company's industry.\n"
)


def summarize_chunk(chunk, user_prompt=None, context= True ):
    """Uses OpenAI API to summarize the given chunk."""

    system_prompt = "You are a market research professional. Return your results in Markdown format."
    
    prompt = chunk
    response = openai.ChatCompletion.create(
          #deployment_id=Config.deployment_name,
          model="gpt-3.5-turbo",
          messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": user_prompt}],
          temperature=0.1,
          max_tokens=500,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
    output_summary = response["choices"][0]["message"]["content"]
    return output_summary

def summarize_document(document, search_context= True):
    """Chunks and summarizes the document, then produces a global summary."""
    global_summary = summarize_chunk(document, user_prompt=global_prompt, context= search_context)
    #global_summary = summarize_chunk(document, user_prompt=rewrite_prompt)
    
    return global_summary

# function structure

wiki_info = [
    {
        'name': 'extract_company_info',
        'description': 'Get company details from text',
        'parameters': {
            'type': 'object',
            'properties': {
                'address_details': {
                    'type': 'string',
                    'description': "Adress of company's head quarter"
                },
                'phone': {
                    'type': 'string',
                    'description': 'All phone numbers mentioned of the company'
                },

                
                'emails': {
                    'type': 'string',
                    'description': 'Email adress of the company '
                },
                 
               
              'industry_details': {
                    'type': 'string',
                    'description': 'Industry details of the company'
                },
              
                'webpage': {
                    'type': 'string',
                    'description': 'web page of the company'
                },

              'subsidaries': {
                    'type': 'string',
                    'description': 'names of subsidaries companies or brand'
                },
              'sic_code': {
                    'type': 'integer',
                    'description': 'Industry details of the company'
                }
            }
        }
    }
]

# main function to be called
def get_company_details(company, with_search_context= False):
    
    summary_all_results = summarize_document(company)


    response = openai.ChatCompletion.create(
                #deployment_id=Config.deployment_name,
                model="gpt-3.5-turbo",
                messages = [{'role': 'user', 'content': summary_all_results }],
                functions = wiki_info,
                function_call = 'auto')

    res_dict = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    
    return res_dict, summary_all_results
 
main_prompt = (
    "You are given a list of companies. Categorize them into unique companies, their subsidiaries and duplicates. Final output should be in JSON format with unique companies and their corressponding subsidiaries and duplicates separately in different lists. Please note that I don't need any steps or code. I need the final output with unique companies and their corressponding subsidiaries and duplicates in JSON format after considering rebranding, acquisitions, mergers, etc." 
)

def duplicate_companies(chunk, user_prompt=None, context= True ):
    """Uses OpenAI API to identify duplicate companies from the given list."""

    prompt = chunk
    response = openai.ChatCompletion.create(
          #deployment_id=Config.deployment_name
          model="gpt-4",
          messages=[#{"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": user_prompt}],
          temperature=0.1,
          max_tokens=500,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
    output_summary = response["choices"][0]["message"]["content"]
    return output_summary
      
####### Header #################
st.markdown("""
<center>
<h1 style="font-size: 30px; text-align: center; font-family: 'Raleway'; color : #525252; ">DQM GenAI POC</h1>
</center>
""", unsafe_allow_html=True)

######## Option menu ##########
om1, om2,om3 = st.columns([1,4,1])
with om2:    
    selected_option = option_menu(
        menu_title = None,
        options = ["Harmonization", "Deduplication"],
        icons = ["gear-fill", "table"],
        default_index = 0,
        orientation = "horizontal",
        styles={
        "nav-link-selected": {"background-color": "gray"},
    }
    )

###### Custom info ######
st.markdown(
    """
    <style>
    /* Custom CSS class for the custom st.info widget */
    .custom-st-info {
        background-color: gray; /* Change the background color to gray */
        color: white; /* Change the font color to white */
        padding: 10px 10px; /* Add padding for better visual appearance */
    }
    </style>
    """,
    unsafe_allow_html=True
)

######### Main Background ##########

def main_bg(main_bg):
    main_bg_ext = 'png'
    st.markdown(
      f"""
      <style>
      [data-testid="stAppViewContainer"] > .main {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
        background-size: 180%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
      }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
      </style>
      """,
      unsafe_allow_html=True,
      )
#main_bg_img = 'imgs\warehouse.jpeg'
#main_bg(main_bg_img)

######### Sidebar Background ##############
def sidebar_bg(side_bg):
   side_bg_ext = 'png'
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
#side_bg = 'imgs\pexels-eberhard-grossgasteiger-1743392.jpg'
#sidebar_bg(side_bg)

####### Color change for labels ###########33
st.markdown(
    """
    <style>
    .stNumberInput label {
        color: white; 
    }
    .stSelectbox label {
        color: white; 
    }
    .stMultiSelect label {
        color: white;
    }
    .stFileInput label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        
custom_styles = """
    <style>
    .custom-metric {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .custom-metric .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .custom-metric .metric-label {
        font-size: 12px;
        color: #777;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .custom-metric-ctd{
        background-color: #B3CDCB;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .custom-metric-ctd .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .custom-metric-ctd .metric-label {
        font-size: 12px;
        color: #777;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    .custom-metric-co2{
        background-color: #DBDEF9;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .custom-metric-co2 .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .custom-metric-co2 .metric-label {
        font-size: 12px;
        color: #777;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    </style>
    """
# Render custom CSS styles for metric tiles
st.markdown(custom_styles, unsafe_allow_html=True)   

if selected_option in ('Harmonization'):
    display_df = -1
    with st.sidebar:
        st.title("Harmonization Input")
        line_break(st.sidebar, 1)
        
        selected_company = st.sidebar.text_input('**Company Name**')
        
        if selected_company == '':
            default_ix=0
            custom_info('Please enter the Company Name above')
        else:
            try:
                res_dict, summary_all_results = get_company_details(selected_company, with_search_context= False)
                res_df = pd.json_normalize(res_dict)
                res_df.columns = ['Address','Phone','Email','Details','Subsidaries','SIC Code']
                display_df = 1
                
            except:
                res_dict, summary_all_results = 'Sorry did not get results', 'Sorry did not get results'
                display_df = 0
                
    if display_df == 1:
        st.dataframe(res_df,hide_index=True)
    elif display_df == 0:
        st.text('Sorry no results found for the entered company')
    else:
        st.text('Results of entered Company will be dispalayed here')
        
if selected_option in ('Deduplication'):
    display_df = -1
    
    with st.sidebar:
    
        ############# Upload Button for Company Names ############
        st.title("Deduplication Input")
        uploaded_file = st.file_uploader("Upload an Excel file with Company Names in column A (With no header)", type=["xls", "xlsx", "csv"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file,header=None)
            df.columns = ['Company Names']
            st.write("Uploaded Companies:")
            st.dataframe(df,hide_index=True)
            
            Company_List = df['Company Names'].tolist()
            Company_String = doc_string = ','.join([str(comp) for comp in Company_List])
            
            try:            
                Companies_summary = json.loads(duplicate_companies(Company_String, user_prompt=main_prompt, context= True))
                results_df = pd.json_normalize(Companies_summary['Unique Companies'])
                display_df = 1
            except:
                display_df = 0

    if display_df == 1:
        st.dataframe(results_df,hide_index=True)
    elif display_df == 0:
        st.text('Sorry no results displayed for the uploaded companies')
    else:
        st.text('Results of uploaded Companies will be dispalayed here')
        
hide_footer = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """
st.markdown(hide_footer, unsafe_allow_html=True)