from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import hashlib
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()  # Carga variables del archivo .env

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_TOKEN no configurado en .env")

COHERE_TOKEN = os.getenv("COHERE_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("COHERE_TOKEN no configurado en .env")

GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("GEMINI_TOKEN no configurado en .env")

# API de IAs
import cohere
import google.generativeai as genai
from huggingface_hub import InferenceClient

# Llaves de IAs
co = cohere.Client(COHERE_TOKEN)
genai.configure(api_key=GEMINI_TOKEN)
client = InferenceClient(api_key=HUGGING_FACE_TOKEN)

# Modelos
model_gemini = genai.GenerativeModel("models/gemini-2.5-flash")
model_cohere = "command-r-plus-08-2024"
model_llama = "meta-llama/Llama-3.2-3B-Instruct"
model_mistral = "mistralai/Mistral-7B-Instruct-v0.3"
model_qwen = "Qwen/Qwen2.5-7B-Instruct"

# Historial Chat
cohere_chat_history = []
gemini_chat_history = model_gemini.start_chat(history=[])
qwen_chat_history = []
mistral_chat_history = []
llama_chat_history = []

#Variables Universales
TIMEOUT_LIMIT = 20
VOTING_LIMIT = 6

''' ------------------------------------ '''

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde el HTML

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ruta absoluta de tu HTML (puede estar en la misma carpeta que este script)
HTML_PATH = os.path.join(os.path.dirname(__file__), 'sesion.html')


def crear_engine_sqlserver(BD):
    """Crear conexión a SQL Server"""
    server = r'.\SQLEXPRESS'
    database = BD
    connection_url = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    engine = create_engine(connection_url)
    return engine


def hash_password(password):
    """Hashear la contraseña usando SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/static/graficos/<path:filename>')
def serve_grafico(filename):
    """Sirve los gráficos generados"""
    graficos_path = os.path.join(os.path.dirname(__file__), 'static', 'graficos')
    return send_from_directory(graficos_path, filename)


@app.route('/')
def serve_login_page():
    """Sirve el HTML de login desde cualquier ubicación"""
    return send_file(HTML_PATH)


@app.route('/api/login', methods=['POST'])
def iniciar_sesion():
    """Endpoint para iniciar sesión y verificar credenciales."""
    try:
        datos = request.get_json()
        email = datos.get('email')
        password = datos.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Faltan credenciales (email o password)'}), 400

        contrasena_hash = hash_password(password)

        engine = crear_engine_sqlserver('BD_Medic_AI')
        with engine.connect() as conn:
            query = text("""
                SELECT COUNT(*) 
                FROM Usuarios 
                WHERE email = :email AND contrasena = :contrasena_hash
            """)
            result = conn.execute(query, {"email": email, "contrasena_hash": contrasena_hash}).fetchone()
            count = result[0]

        if count == 1:
            logger.info(f"Inicio de sesión exitoso para: {email}")
            return jsonify({'success': True, 'message': 'Inicio de sesión exitoso'}), 200
        else:
            logger.warning(f"Intento de inicio de sesión fallido para: {email}")
            return jsonify({'success': False, 'message': 'Credenciales inválidas'}), 401

    except Exception as e:
        logger.error(f"Error en iniciar_sesion: {str(e)}")
        return jsonify({'success': False, 'message': 'Error interno del servidor'}), 500


@app.route('/api/verificar-email', methods=['POST'])
def verificar_email():
    """Endpoint para verificar si un email ya existe"""
    try:
        datos = request.get_json()
        email = datos.get('email')

        if not email:
            return jsonify({'error': 'Email requerido'}), 400

        engine = crear_engine_sqlserver('BD_Medic_AI')

        with engine.connect() as conn:
            query = text("SELECT COUNT(*) as count FROM Usuarios WHERE email = :email")
            result = conn.execute(query, {"email": email}).fetchone()
            existe = result[0] > 0

        return jsonify({'existe': existe}), 200

    except Exception as e:
        logger.error(f"Error al verificar email: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/crear-cuenta-completa', methods=['POST'])
def crear_cuenta_completa():
    """
    Endpoint para recibir TODOS los datos del usuario y la contraseña,
    y crear el registro completo en la base de datos.
    """
    try:
        datos = request.get_json()

        # Validar campos requeridos
        campos_requeridos = ['nombre', 'email', 'edad', 'sexo', 'enfermedades', 'habitos', 'contrasena']
        for campo in campos_requeridos:
            if campo not in datos or not datos[campo]:
                return jsonify({'success': False, 'error': f'Falta el campo: {campo}'}), 400

        # Hashear la contraseña
        contrasena_hash = hash_password(datos['contrasena'])

        engine = crear_engine_sqlserver('BD_Medic_AI')
        with engine.connect() as conn:
            query = text("""
                INSERT INTO Usuarios (ID_Tipo, Nombre, email, edad, sexo, enfermedades, habitos, contrasena, Consultas) 
                VALUES (1, :nombre, :email, :edad, :sexo, :enfermedades, :habitos, :contrasena, 0)
            """)

            conn.execute(query, {
                "nombre": datos['nombre'],
                "email": datos['email'],
                "edad": int(datos['edad']),
                "sexo": datos['sexo'],
                "enfermedades": datos['enfermedades'],
                "habitos": datos['habitos'],
                "contrasena": contrasena_hash,
            })
            conn.commit()

        logger.info(f"Cuenta creada y verificada exitosamente para: {datos['email']}")
        return jsonify({'success': True, 'message': 'Cuenta creada exitosamente'}), 201

    except Exception as e:
        logger.error(f"Error al crear cuenta completa: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/get-user-data', methods=['POST'])
def get_user_data():
    """Devuelve el nombre de un usuario a partir de su email."""
    try:
        datos = request.get_json()
        email = datos.get('email')
        if not email:
            return jsonify({'success': False, 'error': 'Falta el email'}), 400

        engine = crear_engine_sqlserver('BD_Medic_AI')
        with engine.connect() as conn:
            query = text("SELECT Nombre FROM Usuarios WHERE email = :email")
            result = conn.execute(query, {"email": email}).fetchone()

        if result:
            nombre_completo = result[0]
            return jsonify({'success': True, 'nombre': nombre_completo}), 200
        else:
            return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404

    except Exception as e:
        logger.error(f"Error en get_user_data: {str(e)}")
        return jsonify({'success': False, 'error': 'Error interno del servidor'}), 500


@app.route('/api/usuarios', methods=['GET'])
def obtener_usuarios():
    """Endpoint para obtener todos los usuarios (opcional, para testing)"""
    try:
        engine = crear_engine_sqlserver('BD_Medic_AI')

        with engine.connect() as conn:
            query = text("SELECT * FROM Usuarios")
            result = conn.execute(query)
            usuarios = []
            for row in result:
                usuarios.append({
                    'IdUsuario': row[0] if len(row) > 0 else None,
                    'IdTipo': row[1] if len(row) > 1 else None,
                    'nombre': row[2] if len(row) > 2 else None,
                    'email': row[3] if len(row) > 3 else None,
                    'edad': row[4] if len(row) > 4 else None,
                    'sexo': row[5] if len(row) > 5 else None,
                    'enfermedades': row[6] if len(row) > 6 else None,
                    'habitos': row[7] if len(row) > 7 else None
                })

        return jsonify({'usuarios': usuarios}), 200

    except Exception as e:
        logger.error(f"Error al obtener usuarios: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que el servidor está corriendo"""
    return jsonify({'status': 'OK', 'message': 'Servidor funcionando correctamente'}), 200

def verificar_y_resetear_diario():
    """Verifica si es un nuevo día y resetea Consultas automáticamente"""
    try:
        engine = crear_engine_sqlserver('BD_Medic_AI')
        with engine.connect() as conn:
            conn.execute(text("EXEC sp_VerificarYResetear"))
            conn.commit()
    except Exception as e:
        logger.error(f"Error al verificar reset diario: {str(e)}")

def extraer_numero_voto(texto):
    """Extrae el primer número encontrado en el texto"""
    import re
    numeros = re.findall(r'\d+', str(texto))
    if numeros:
        num = int(numeros[0])
        if 1 <= num <= 5:
            return num
    return None

def insertar_respuesta(id_ia, respuesta, tiempo):
    # Insertar relacion de sintomas con prompt a Tabla "DetallesPrompt "
    engine = crear_engine_sqlserver('BD_Medic_AI')
    with engine.connect() as conn:
        # Obtener el IDPrompt
        query = text(""" SELECT TOP 1 IDPrompt
                                            FROM Prompts 
                                            ORDER BY Fecha DESC
                                        """)
        resultado = conn.execute(query).fetchone()
        id_prompt = resultado[0]

        query = text("""
                                        INSERT INTO Respuestas (IDPrompt, ID_IA, Respuesta, Tiempo) 
                                        VALUES (:id_prompt, :id_ia, :respuesta, :tiempo)
                                    """)

        conn.execute(query, {
            "id_prompt": id_prompt,
            "id_ia": id_ia,
            "respuesta": respuesta,
            "tiempo": tiempo
        })
        conn.commit()


class Cronometro:
    def __init__(self):
        self.tiempos = {}

    def iniciar(self, nombre):
        self.tiempos[nombre] = {'inicio': time.perf_counter()}

    def detener(self, nombre):
        if nombre in self.tiempos:
            fin = time.perf_counter()
            inicio = self.tiempos[nombre]['inicio']
            self.tiempos[nombre]['duracion'] = fin - inicio
            return self.tiempos[nombre]['duracion']

    def imprimir_tiempo(self, nombre):
        """Imprime el tiempo de un cronómetro específico"""
        if nombre in self.tiempos and 'duracion' in self.tiempos[nombre]:
            tiempo = self.tiempos[nombre]['duracion']
            print(f"⏱️ {nombre}: {tiempo:.2f} segundos")
            return tiempo
        else:
            print(f"⚠️ No se encontró el cronómetro '{nombre}'")
            return None

    def extraer_tiempo(self, nombre):
        """Imprime el tiempo de un cronómetro específico"""
        if nombre in self.tiempos and 'duracion' in self.tiempos[nombre]:
            tiempo = self.tiempos[nombre]['duracion']
            return tiempo
        else:
            return None

class RespuestaError:
    def __init__(self, text):
        self.text = text

@app.route('/api/send-prompt', methods=['POST'])
def send_prompt():
    """
    Recibe los síntomas, busca los datos del usuario en la BD,
    construye el prompt completo y lo devuelve.
    """
    verificar_y_resetear_diario()

    cronometro = Cronometro()

    cronometro.iniciar('full_time')

    try:
        datos_entrada = request.get_json()

        sintomas = datos_entrada.get('sintomas')
        email = datos_entrada.get('email')

        print("==================================================")
        print("Datos de entrada recibidos:", datos_entrada)
        print("Síntomas extraídos:", sintomas)
        print("Email extraído:", email)
        print("==================================================")

        if not sintomas or not email:
            cronometro.detener('full_time')
            cronometro.imprimir_tiempo('full_time')
            return jsonify({'success': False, 'error': 'Faltan síntomas o email'}), 400

        # Ingresar Prompt a Tabla "Prompts "
        engine = crear_engine_sqlserver('BD_Medic_AI')
        with engine.connect() as conn:
            # Obtener el numero de consultas
            query = text("SELECT Consultas FROM Usuarios WHERE email = :email")
            consultas_result = conn.execute(query, {"email": email}).fetchone()
            num_consultas = consultas_result[0]

            print("Num Consultas: ", num_consultas)

        if num_consultas < 10:

            # Verificar si hay relacion con PalabrasClave
            engine = crear_engine_sqlserver('BD_Medic_AI')
            with engine.connect() as conn:

                tabla = "Sintomas"
                columna = "PalabrasClave"
                palabras = re.findall(r'\b\w+\b', sintomas.lower())

                # Construir consulta con parámetros NOMBRADOS (:param0, :param1, etc.)
                condiciones = " OR ".join([f"{columna} LIKE :param{i}" for i in range(len(palabras))])
                query_str = f"SELECT IDSintoma, {columna} FROM {tabla} WHERE {condiciones}"

                # Preparar parámetros como DICCIONARIO
                params_dict = {f'param{i}': f'%{palabra}%' for i, palabra in enumerate(palabras)}

                # SOLUCIÓN: capturar el resultado y usar text()
                result = conn.execute(text(query_str), params_dict)
                resultados = result.fetchall()

                keys = [key[0] for key in resultados]

            if len(resultados) > 0:

                tie_time = {}
                TIMEOUTS = 0

                #Ingresar Prompt a Tabla "Prompts "
                engine = crear_engine_sqlserver('BD_Medic_AI')
                with engine.connect() as conn:


                    # Obtener el IDUsuario
                    query = text("SELECT IDUsuario FROM Usuarios WHERE email = :email")
                    id_result = conn.execute(query, {"email": email}).fetchone()
                    id_usuario = id_result[0]

                    query = text("""
                                INSERT INTO Prompts (IDUsuario, Contenido, Fecha, ID_IA, Gravedad) 
                                VALUES (:id_usuario, :contenido, GETDATE(), :id, :gravedad)
                            """)

                    conn.execute(query, {
                        "id_usuario": id_usuario,
                        "contenido": sintomas,
                        "id": -1,
                        "gravedad": "Pendiente"
                    })
                    conn.commit()

                # Insertar relacion de sintomas con prompt a Tabla "DetallesPrompt "
                engine = crear_engine_sqlserver('BD_Medic_AI')
                with engine.connect() as conn:
                    # Obtener el IDPrompt
                    query = text(""" SELECT TOP 1 IDPrompt
                                        FROM Prompts 
                                        ORDER BY Fecha DESC
                                    """)
                    resultado = conn.execute(query).fetchone()
                    id_prompt = resultado[0]

                    query = text("""
                                    INSERT INTO DetallesPrompt (IDPrompt, IDSintoma) 
                                    VALUES (:id_prompt, :id_sintoma)
                                """)
                    for key in keys:
                        conn.execute(query, {
                            "id_prompt": id_prompt,
                            "id_sintoma": key
                        })
                    conn.commit()


                engine = crear_engine_sqlserver('BD_Medic_AI')
                with engine.connect() as conn:
                    query = text("SELECT edad, sexo, enfermedades, habitos FROM Usuarios WHERE email = :email")
                    usuario = conn.execute(query, {"email": email}).fetchone()

                if not usuario:
                    cronometro.detener('full_time')
                    cronometro.imprimir_tiempo('full_time')

                    return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404

                edad, sexo, enfermedades, habitos = usuario

                prompt_completo = f"""Rol/Entorno: "Actúa como un médico general especializado en diagnóstico inicial de síntomas comunes."
                                        Tarea: "Analiza los siguientes síntomas del usuario: [{sintomas}]."
                                        Resultado deseado: "Devuelve el diagnóstico en formato de texto estructurado con las secciones: diagnóstico, gravedad (leve, moderada, grave), explicación y recomendaciones médicas."
                                        Parámetros: "Usa lenguaje claro y comprensible para un paciente no especializado, sin mencionar medicamentos ni tratamientos."
                                        Receptor: "Paciente adulto con conocimiento básico en salud. Información relevante del usuario (edad: {edad}, sexo: {sexo}, enfermedades crónicas: {enfermedades}, hábitos: {habitos}) registrada en la base de datos."
                                        """


                # COHERE con timeout manual
                cronometro.iniciar('cohere_time')

                def llamada_cohere():
                    return co.chat(
                        model=model_cohere,
                        message=prompt_completo,
                        chat_history=cohere_chat_history
                    )

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(llamada_cohere)
                        chat_response_cohere = future.result(timeout=TIMEOUT_LIMIT)

                    cohere_chat_history.append({"role": "USER", "message": prompt_completo})
                    cohere_chat_history.append({"role": "CHATBOT", "message": chat_response_cohere.text})
                    cronometro.detener('cohere_time')
                    cronometro.imprimir_tiempo('cohere_time')

                    cohere_tiempo = cronometro.extraer_tiempo('cohere_time')
                    tie_time[1] = cohere_tiempo
                    insertar_respuesta(1, chat_response_cohere.text, cohere_tiempo)

                except TimeoutError:
                    cronometro.detener('cohere_time')
                    print(f"""⚠️ ERROR: Cohere excedió el límite de {TIMEOUT_LIMIT} segundos""")
                    tie_time[1] = None
                    insertar_respuesta(1, "TIMEOUT", None)
                    chat_response_cohere = RespuestaError("TIMEOUT")
                    TIMEOUTS += 1

                except Exception as e:
                    cronometro.detener('cohere_time')
                    print(f"⚠️ ERROR Cohere: {str(e)}")
                    tie_time[1] = None
                    insertar_respuesta(1, f"ERROR: {str(e)}", None)
                    chat_response_cohere = RespuestaError("ERROR")
                    TIMEOUTS += 1


                # GEMINI con timeout manual
                cronometro.iniciar('gemini_time')

                def llamada_gemini():
                    return gemini_chat_history.send_message(prompt_completo)

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(llamada_gemini)
                        chat_response_gemini = future.result(timeout=TIMEOUT_LIMIT)

                    cronometro.detener('gemini_time')
                    cronometro.imprimir_tiempo('gemini_time')

                    gemini_tiempo = cronometro.extraer_tiempo('gemini_time')
                    tie_time[2] = gemini_tiempo
                    insertar_respuesta(2, chat_response_gemini.text, gemini_tiempo)

                except TimeoutError:
                    cronometro.detener('gemini_time')
                    print(f"""⚠️ ERROR: Gemini excedió el límite de {TIMEOUT_LIMIT} segundos""")
                    tie_time[2] = None
                    insertar_respuesta(2, "TIMEOUT", None)
                    chat_response_gemini = RespuestaError("TIMEOUT")
                    TIMEOUTS += 1


                except Exception as e:
                    cronometro.detener('gemini_time')
                    print(f"⚠️ ERROR Gemini: {str(e)}")
                    tie_time[2] = None
                    insertar_respuesta(2, f"ERROR: {str(e)}", None)
                    chat_response_gemini = RespuestaError("ERROR")
                    TIMEOUTS += 1



                # QWEN (Hugging Face)
                cronometro.iniciar('qwen_time')

                def llamada_qwen():
                    qwen_chat_history.append({
                        "role": "user",
                        "content": prompt_completo
                    })

                    response = client.chat_completion(
                        model=model_qwen,
                        messages=qwen_chat_history,
                        max_tokens=500
                    )
                    return response

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(llamada_qwen)
                        response = future.result(timeout=TIMEOUT_LIMIT)

                    chat_response_qwen = response.choices[0].message.content

                    qwen_chat_history.append({
                        "role": "assistant",
                        "content": chat_response_qwen
                    })

                    cronometro.detener('qwen_time')
                    cronometro.imprimir_tiempo('qwen_time')

                    qwen_tiempo = cronometro.extraer_tiempo('qwen_time')
                    tie_time[3] = qwen_tiempo
                    insertar_respuesta(3, chat_response_qwen, qwen_tiempo)

                except TimeoutError:
                    cronometro.detener('qwen_time')
                    print(f"""⚠️ ERROR: Qwen excedió el límite de {TIMEOUT_LIMIT} segundos""")
                    tie_time[3] = None
                    insertar_respuesta(3, "TIMEOUT", None)
                    chat_response_qwen = RespuestaError("TIMEOUT")
                    TIMEOUTS += 1


                except Exception as e:
                    cronometro.detener('qwen_time')
                    print(f"❌ ERROR Qwen: {str(e)}")
                    tie_time[3] = None
                    insertar_respuesta(3, f"ERROR: {str(e)}", None)
                    chat_response_qwen = RespuestaError("ERROR")
                    TIMEOUTS += 1


                # MISTRAL
                cronometro.iniciar('mistral_time')

                def llamada_mistral():
                    mistral_chat_history.append({
                        "role": "user",
                        "content": prompt_completo
                    })

                    response = client.chat_completion(
                        model=model_mistral,
                        messages=mistral_chat_history,
                        max_tokens=500
                    )
                    return response

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(llamada_mistral)
                        response = future.result(timeout=TIMEOUT_LIMIT)

                    chat_response_mistral = response.choices[0].message.content

                    mistral_chat_history.append({
                        "role": "assistant",
                        "content": chat_response_mistral
                    })

                    cronometro.detener('mistral_time')
                    cronometro.imprimir_tiempo('mistral_time')

                    mistral_tiempo = cronometro.extraer_tiempo('mistral_time')
                    tie_time[4] = mistral_tiempo
                    insertar_respuesta(4, chat_response_mistral, mistral_tiempo)

                except TimeoutError:
                    cronometro.detener('mistral_time')
                    print(f"""⚠️ ERROR: Mistral excedió el límite de {TIMEOUT_LIMIT} segundos""")
                    tie_time[4] = None
                    insertar_respuesta(4, "TIMEOUT", None)
                    chat_response_mistral = RespuestaError("TIMEOUT")

                    TIMEOUTS += 1


                except Exception as e:
                    cronometro.detener('mistral_time')
                    print(f"❌ ERROR Mistral: {str(e)}")
                    tie_time[4] = None
                    insertar_respuesta(4, f"ERROR: {str(e)}", None)
                    chat_response_mistral = RespuestaError("ERROR")
                    TIMEOUTS += 1


                # LLAMA
                cronometro.iniciar('llama_time')

                def llamada_llama():
                    llama_chat_history.append({
                        "role": "user",
                        "content": prompt_completo
                    })

                    response = client.chat_completion(
                        model=model_llama,
                        messages=llama_chat_history,
                        max_tokens=500
                    )
                    return response

                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(llamada_llama)
                        response = future.result(timeout=TIMEOUT_LIMIT)

                    chat_response_llama = response.choices[0].message.content

                    llama_chat_history.append({
                        "role": "assistant",
                        "content": chat_response_llama
                    })

                    cronometro.detener('llama_time')
                    cronometro.imprimir_tiempo('llama_time')

                    llama_tiempo = cronometro.extraer_tiempo('llama_time')
                    tie_time[5] = llama_tiempo
                    insertar_respuesta(5, chat_response_llama, llama_tiempo)

                except TimeoutError:
                    cronometro.detener('llama_time')
                    print(f"""⚠️ ERROR: Llama excedió el límite de {TIMEOUT_LIMIT} segundos""")
                    tie_time[5] = None
                    insertar_respuesta(5, "TIMEOUT", None)
                    chat_response_llama = RespuestaError("TIMEOUT")
                    TIMEOUTS += 1


                except Exception as e:
                    cronometro.detener('llama_time')
                    print(f"❌ ERROR Llama: {str(e)}")
                    tie_time[5] = None
                    insertar_respuesta(5, f"ERROR: {str(e)}", None)
                    chat_response_llama = RespuestaError("ERROR")
                    TIMEOUTS += 1

                if TIMEOUTS < 3:

                    respuestas_ias = {1: chat_response_cohere.text, 2: chat_response_gemini.text, 3: chat_response_qwen, 4: chat_response_mistral, 5: chat_response_llama}
                    votos = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

                    # Modelo Meta-Votacion / 1:COHERE / 2:GEMINI / 3:QWEN / 4:MISTRAL / 5:LLAMA /
                    prompt_votacion = f"""Rol/Entorno: "Actúa como un médico general el cual va elegir entre multiples diagnosticos el mejor."
                                                    Tarea: "Analiza los siguientes diagnosticos numerados, en caso de que un resultado sea "TIMEOUT" o "ERROR" no lo tomes en cuenta al momento de la votacion: 1:[{chat_response_cohere.text}], 2:[{chat_response_gemini.text}], 3:[{chat_response_qwen}], 4:[{chat_response_mistral}], 5:[{chat_response_llama}]."
                                                    Resultado deseado: "Devuelve unicamente el numero del diagnostico que elegiste, no agregues contexto ni informacion adicional, unicamente el numero del diagnostico elegido."
                                                    """

                    print("=" * 50)
                    print("Modelo Meta-Votacion / 1:COHERE / 2:GEMINI / 3:QWEN / 4:MISTRAL / 5:LLAMA /")

                    cronometro.iniciar('voting_time')

                    # COHERE
                    votacion_cohere = co.chat(
                        model=model_cohere,
                        message=prompt_votacion,
                        chat_history=cohere_chat_history
                    )

                    print("COHERE Votacion: ", votacion_cohere.text)
                    cohere_chat_history.append({"role": "USER", "message": prompt_votacion})
                    cohere_chat_history.append({"role": "CHATBOT", "message": votacion_cohere.text})

                    voto_cohere = extraer_numero_voto(votacion_cohere.text)
                    if voto_cohere:
                        votos[voto_cohere] += 1
                    else:
                        print(f"⚠️ COHERE voto inválido: {votacion_cohere.text}")
                    # Fin COHERE

                    # GEMINI
                    votacion_gemini = gemini_chat_history.send_message(prompt_votacion)
                    print("GEMINI Votacion: ", votacion_gemini.text)

                    voto_gemini = extraer_numero_voto(votacion_gemini.text)
                    if voto_gemini:
                        votos[voto_gemini] += 1
                    else:
                        print(f"⚠️ GEMINI voto inválido: {votacion_gemini.text}")
                    # Fin GEMINI

                    # QWEN
                    def llamada_qwen():
                        qwen_chat_history.append({
                            "role": "user",
                            "content": prompt_votacion
                        })

                        response = client.chat_completion(
                            model=model_qwen,
                            messages=qwen_chat_history,
                            max_tokens=500
                        )
                        return response

                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(llamada_qwen)
                            response = future.result(timeout=VOTING_LIMIT)

                        votacion_qwen = response.choices[0].message.content

                        qwen_chat_history.append({
                            "role": "assistant",
                            "content": votacion_qwen
                        })

                        print("Qwen Votacion: ", votacion_qwen)
                        voto_qwen = extraer_numero_voto(votacion_qwen)
                        if voto_qwen:
                            votos[voto_qwen] += 1
                        else:
                            print(f"⚠️ Qwen voto inválido: {votacion_qwen}")

                    except TimeoutError:
                        print(f"""⚠️ ERROR: Qwen excedió el límite de {VOTING_LIMIT} segundos""")

                    except Exception as e:
                        print(f"❌ ERROR Qwen: {str(e)}")


                    #MISTRAL
                    def llamada_mistral():
                        mistral_chat_history.append({
                            "role": "user",
                            "content": prompt_votacion
                        })

                        response = client.chat_completion(
                            model=model_mistral,
                            messages=mistral_chat_history,
                            max_tokens=500
                        )
                        return response

                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(llamada_mistral)
                            response = future.result(timeout=VOTING_LIMIT)

                        votacion_mistral = response.choices[0].message.content

                        mistral_chat_history.append({
                            "role": "assistant",
                            "content": votacion_mistral
                        })

                        print("Mistral Votacion: ", votacion_mistral)
                        voto_mistral = extraer_numero_voto(votacion_mistral)
                        if voto_mistral:
                            votos[voto_mistral] += 1
                        else:
                            print(f"⚠️ Mistral voto inválido: {votacion_mistral}")

                    except TimeoutError:
                        print(f"""⚠️ ERROR: Mistral excedió el límite de {VOTING_LIMIT} segundos""")

                    except Exception as e:
                        print(f"❌ ERROR Mistral: {str(e)}")

                    # LLAMA

                    def llamada_llama():
                        llama_chat_history.append({
                            "role": "user",
                            "content": prompt_votacion
                        })

                        response = client.chat_completion(
                            model=model_llama,
                            messages=llama_chat_history,
                            max_tokens=500
                        )
                        return response

                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(llamada_llama)
                            response = future.result(timeout=VOTING_LIMIT)

                        votacion_llama = response.choices[0].message.content

                        llama_chat_history.append({
                            "role": "assistant",
                            "content": votacion_llama
                        })

                        print("LLAMA Votacion: ", votacion_llama)
                        voto_llama = extraer_numero_voto(votacion_llama)
                        if voto_llama:
                            votos[voto_llama] += 1
                        else:
                            print(f"⚠️ LLAMA voto inválido: {votacion_llama}")

                    except TimeoutError:
                        print(f"""⚠️ ERROR: Llama excedió el límite de {VOTING_LIMIT} segundos""")

                    except Exception as e:
                        print(f"❌ ERROR Llama: {str(e)}")



                    print("=" * 50)

                    print(votos)
                    max_valor = max(votos.values())
                    # top_votos = max(votos, key=votos.get)
                    top_votos = [k for k, v in votos.items() if v == max_valor]

                    ia_seleccionada = None

                    if len(top_votos) > 1:

                        least_time = float('inf')

                        for ia in top_votos:
                            if tie_time[ia] < least_time:
                                least_time = tie_time[ia]
                                ia_seleccionada = ia
                    else:
                        ia_seleccionada = top_votos[0]

                    print("GANO LA IA: ", ia_seleccionada)

                    prompt_prelim = respuestas_ias[ia_seleccionada]
                    prompt_seleccionado = prompt_prelim.replace('#', '').replace('*', '')
                    cronometro.detener('voting_time')
                    cronometro.imprimir_tiempo('voting_time')

                    logger.info(f"Prompt completo generado para {email}")

                    # Insertar IA Ganadora
                    engine = crear_engine_sqlserver('BD_Medic_AI')
                    with engine.connect() as conn:
                        # Obtener el IDPrompt
                        query = text(""" SELECT TOP 1 IDPrompt
                                                            FROM Prompts 
                                                            ORDER BY Fecha DESC
                                                        """)
                        resultado = conn.execute(query).fetchone()
                        id_prompt = resultado[0]

                        query = text("""
                                                        UPDATE Prompts  
                                                        SET ID_IA = :ia_ganadora
                                                        WHERE IDPrompt = :id_prompt
                                                    """)

                        conn.execute(query, {
                            "ia_ganadora": ia_seleccionada,
                            "id_prompt": id_prompt
                        })
                        conn.commit()

                    # Extraer nivel de gravedad

                    leve = "leve"
                    moderada = "moderada"
                    grave = "grave"

                    conteo_leve = prompt_seleccionado.lower().count(leve.lower())
                    conteo_moderada = prompt_seleccionado.lower().count(moderada.lower())
                    conteo_grave = prompt_seleccionado.lower().count(grave.lower())

                    if all([conteo_leve == 0, conteo_moderada == 0, conteo_grave == 0]):
                        print("Gravedad: No se encontro un nivel de gravedad")

                        # Insertar nivel de gravedad
                        engine = crear_engine_sqlserver('BD_Medic_AI')
                        with engine.connect() as conn:
                            # Obtener el IDPrompt
                            query = text(""" SELECT TOP 1 IDPrompt
                                                                                                            FROM Prompts 
                                                                                                            ORDER BY Fecha DESC
                                                                                                        """)
                            resultado = conn.execute(query).fetchone()
                            id_prompt = resultado[0]

                            query = text("""
                                                                                                        UPDATE Prompts  
                                                                                                        SET Gravedad = 'NO FIND'
                                                                                                        WHERE IDPrompt = :id_prompt
                                                                                                    """)

                            conn.execute(query, {
                                "id_prompt": id_prompt
                            })
                            conn.commit()


                    else:

                        max_nivel = max(conteo_leve, conteo_moderada, conteo_grave)

                        gravedad = ""
                        if conteo_leve == max_nivel:
                            gravedad += leve
                            gravedad += " "

                        if conteo_moderada == max_nivel:
                            gravedad += moderada
                            gravedad += " "

                        if conteo_grave == max_nivel:
                            gravedad += grave
                            gravedad += " "

                        print("Gravedad: ", gravedad)

                        # Insertar nivel de gravedad
                        engine = crear_engine_sqlserver('BD_Medic_AI')
                        with engine.connect() as conn:
                            # Obtener el IDPrompt
                            query = text(""" SELECT TOP 1 IDPrompt
                                                                                    FROM Prompts 
                                                                                    ORDER BY Fecha DESC
                                                                                """)
                            resultado = conn.execute(query).fetchone()
                            id_prompt = resultado[0]

                            query = text("""
                                                                                UPDATE Prompts  
                                                                                SET Gravedad = :gravedad
                                                                                WHERE IDPrompt = :id_prompt
                                                                            """)

                            conn.execute(query, {
                                "gravedad": gravedad,
                                "id_prompt": id_prompt
                            })
                            conn.commit()

                            #Grafico Visual Gravedad

                            # Datos originales
                            categorias_completas = ['Leve', 'Moderada', 'Grave']
                            valores_completos = [conteo_leve, conteo_moderada, conteo_grave]
                            colores_completos = ['#77bd66', '#f9e547', '#f75c4c']

                            # Filtrar solo los valores mayores a 0
                            categorias = []
                            valores = []
                            colores = []

                            for i in range(len(valores_completos)):
                                if valores_completos[i] > 0:
                                    categorias.append(categorias_completas[i])
                                    valores.append(valores_completos[i])
                                    colores.append(colores_completos[i])

                            # Solo crear el gráfico si hay al menos un valor mayor a 0
                            if len(valores) > 0:
                                """plt.figure(figsize=(6, 6))
                                plt.pie(valores, labels=categorias, colors=colores, autopct='%1.1f%%', startangle=90)
                                plt.title('Nivel de Gravedad Detectado')

                                # Guardar el gráfico como imagen
                                img_path = os.path.join(os.path.dirname(__file__), 'static', 'graficos')
                                os.makedirs(img_path, exist_ok=True)
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                img_filename = f'gravedad_{timestamp}.png'
                                img_fullpath = os.path.join(img_path, img_filename)
                                plt.savefig(img_fullpath, bbox_inches='tight', dpi=100)
                                plt.close()"""

                                fig, ax = plt.subplots(figsize=(6, 6))

                                # Cambiar el color de fondo de la figura completa
                                fig.patch.set_facecolor('#7c3aad')  # Fondo morado como tu diseño

                                # Cambiar el color de fondo del área del gráfico
                                ax.set_facecolor('#7c3aad')  # Mismo color morado

                                # Crear el gráfico de pastel
                                ax.pie(valores, labels=categorias, colors=colores, autopct='%1.1f%%', startangle=90)

                                # Cambiar color del título
                                ax.set_title('Nivel de Gravedad Detectado', color='white', fontsize=14,
                                             fontweight='bold')

                                # Guardar el gráfico como imagen
                                img_path = os.path.join(os.path.dirname(__file__), 'static', 'graficos')
                                os.makedirs(img_path, exist_ok=True)
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                img_filename = f'gravedad_{timestamp}.png'
                                img_fullpath = os.path.join(img_path, img_filename)
                                plt.savefig(img_fullpath, bbox_inches='tight', dpi=100, facecolor='#7c3aad')
                                plt.close()

                                # URL relativa para enviar al frontend
                                img_url = f'/static/graficos/{img_filename}'
                            else:
                                # Si todos los valores son 0, no generar gráfico
                                img_url = None
                                print("⚠️ Todos los valores de gravedad son 0, no se genera gráfico")


                    # Aumentar contador consultas
                    engine = crear_engine_sqlserver('BD_Medic_AI')
                    with engine.connect() as conn:

                        query = text("UPDATE Usuarios SET Consultas += 1 WHERE email = :email")
                        conn.execute(query, {"email": email})
                        conn.commit()

                    cronometro.detener('full_time')
                    cronometro.imprimir_tiempo('full_time')

                    return jsonify({
                        'success': True,
                        'respuesta': prompt_seleccionado,
                        'grafico_url': img_url,
                        'gravedad': gravedad.strip()
                    }), 200

                else:
                    cronometro.detener('full_time')
                    cronometro.imprimir_tiempo('full_time')

                    return jsonify({'success': True,
                                    'respuesta': "Ocurrio un error al procesar tu consulta. Intentalo nuevamente"}), 200


            else:
                cronometro.detener('full_time')
                cronometro.imprimir_tiempo('full_time')

                # No hay coincidencias con enfermedades respiratorias
                return jsonify({
                    'success': True,
                    'prompt_invalido': True,
                    'respuesta': "Lo siento, solo puedo ayudar con síntomas relacionados a enfermedades respiratorias."
                }), 200
        else:
            cronometro.detener('full_time')
            cronometro.imprimir_tiempo('full_time')

            return jsonify({'success': True,
                            'respuesta': "Has alcanzado el numero maximo de consultas hoy. Intentalo nuevamente mañana"}), 200

    except Exception as e:
        logger.error(f"Error en send_prompt: {str(e)}")
        cronometro.detener('full_time')
        cronometro.imprimir_tiempo('full_time')

        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)