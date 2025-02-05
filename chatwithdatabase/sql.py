import sqlite3

conn = sqlite3.connect("student.db")

cursor = conn.cursor()

tableinfo = """
  create table STUDENT(
    name varchar(25), 
    class varchar(25),
    section varchar(25),
    marks integer
  )
"""

cursor.execute(tableinfo)


cursor.execute(
  """
  INSERT INTO STUDENT (name, class, section, marks)  
    VALUES  
    ('John Doe', 'Tennis', 'A', 85),  
    ('Jane Smith', 'Tennis', 'B', 90),  
    ('David Johnson', 'Tennis', 'A', 78),  
    ('Emily Brown', 'Tennis', 'C', 88);  

"""
)

print("The inserted records are: ")
data = cursor.execute('''select * from STUDENT''')

for row in data:
  print(row)


conn.commit()
conn.close()

