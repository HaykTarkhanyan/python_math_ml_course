# !pip install mysql-connector-python
import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="yourusername",
    password="yourpassword",
    database="mydatabase"   # Optional, use if the database already exists.
)

#####
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE mydatabase")

#####
mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")

#####
sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
mycursor.execute(sql, val)
mydb.commit()


####
mycursor.execute("SELECT * FROM customers")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)


####
sql = "UPDATE customers SET address = 'Canyon 123' WHERE name = 'John'"
mycursor.execute(sql)
mydb.commit()


#### 
sql = "DELETE FROM customers WHERE address = 'Mountain 21'"
mycursor.execute(sql)
mydb.commit()


####
mydb.close()
