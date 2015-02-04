"""test to connect to TestStress"""
from McNulty_config import connect
import pandas.io.sql as psql


db = connect()
sql = "SELECT * FROM all_hospitals"
df = psql.read_sql(sql, db)
print len(df)
# cursor = db.cursor()

# cursor.execute('SHOW TABLES')
# cursor.execute("SELECT * FROM va_tidy;")

# for row in cursor.fetchall():
#     print row

# cursor.close()
db.close()
