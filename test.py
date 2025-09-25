from sqlalchemy import create_engine, text
import pandas as pd

def crear_engine_sqlserver(BD):
    # Parámetros de conexión
    server = r'.\SQLEXPRESS'
    database = BD

    # Crear la URL de conexión
    connection_url = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'

    # Crear el engine
    engine = create_engine(connection_url)
    return engine


# Usar sin warnings
try:
    engine = crear_engine_sqlserver('BD_Medic_AI')

    # Ver tablas disponibles
    df = pd.read_sql("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
    """, engine)

    print("Tablas disponibles:")
    print(df)

except Exception as e:
    print(f"Error: {e}")

try:
    engine = crear_engine_sqlserver('BD_Medic_AI')

    # Ver tablas disponibles
    df = pd.read_sql("""
        SELECT * 
        FROM Usuarios
    """, engine)

    print("Data Info:")
    print(df)

except Exception as e:
    print(f"Error: {e}")

try:
    engine = crear_engine_sqlserver('BD_Medic_AI')

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO Usuarios (Nombre, email) 
            VALUES (:nombre, :email)
        """), {
            "nombre": "Juan Pérez",
            "email": "juan@email.com"
        })
        conn.commit()

except Exception as e:
    print(f"Error: {e}")

try:
    engine = crear_engine_sqlserver('BD_Medic_AI')

    # Ver tablas disponibles
    df = pd.read_sql("""
        SELECT * 
        FROM Usuarios
    """, engine)

    print("Data Info:")
    print(df)

except Exception as e:
    print(f"Error: {e}")
