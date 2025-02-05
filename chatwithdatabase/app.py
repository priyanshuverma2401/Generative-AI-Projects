from dotenv import load_dotenv
import streamlit as st
import os
import sqlite3
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv() #load all enviroment variable

#configure api key
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")


# Define prompt template




#function to load google gemini model and provide sql query as output

def get_gemini_response(question, prompt):
  model = genai.GenerativeModel('gemini-pro')
  response = model.generate_content([prompt, question])
  return response.text

# function to retrive query from the sql database

def read_sql_query(sql, db):
  conn = sqlite3.connect(db)
  cur = conn.cursor()
  cur.execute(sql)
  rows = cur.fetchall()
  conn.commit()
  conn.close()
  for row in rows:
    print(row)
  return rows

# define prompt

prompt = '''
  you're an expert i converting English into sql query!
  the sql database has the name STUDENT and has the following columns - NAME, CLASS, SECTION, MARKS\n\n
  for example
  Example 1: How many entries of records are present?
  the sql cammand will be something like this:
  SELECT COUNT(*) FROM STUDENT;
  Example-2: tell me all the student studying in the data science class?
  the sql cammand should be something like this
  SELECT * FROM STUDENT WHERE CLASS = "Data Science";
  alse  the sql code should not have ``` in the begning or at the end of the word in the output. please don't do any formating only generate sql query don't write ```sql or anything.
'''

def format_prompt(question):
    # Create a prompt that instructs the model to respond in full sentences
    prompt = f"These are the data fetched by sql query related to user question plese make a full sentence answer from this {data}"
    return prompt

## Streamlit app
st.set_page_config(page_title="I can retrieve any SQL Query")
st.header("SQLMate App")

question = st.text_input("Enter your query here: ")
submit = st.button("Ask & Get")

## if submit is clicked

if submit:
  response  = get_gemini_response(question, prompt)
  print(response)
  data  = read_sql_query(response, "student.db")
  prompt2 = format_prompt(str(list(data)))
  print(str(list(data)))
  res2 = get_gemini_response(prompt2, question)
  # st.subheader("The responce is: ")
  # # st.text(response)
  st.text(res2)
  # 

  with st.expander("View Sql Query"):
    st.text(response)





# data = fetch_data()
#     prompt = format_prompt(data, user_input)
#     response = llm.generate(prompt)  # Assuming llm is your language model instance
#     st.text(response)


