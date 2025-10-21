from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
import hashlib
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

@app.route('/api/send-prompt', methods=['POST'])
def send_prompt():
    """
    Recibe los síntomas, busca los datos del usuario en la BD,
    construye el prompt completo y lo devuelve.
    """
    verificar_y_resetear_diario()

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

                # Aumentar contador consultas
                engine = crear_engine_sqlserver('BD_Medic_AI')
                with engine.connect() as conn:

                    query = text("UPDATE Usuarios SET Consultas += 1 WHERE email = :email")
                    conn.execute(query, {"email": email})
                    conn.commit()

                #Ingresar Prompt a Tabla "Prompts "
                engine = crear_engine_sqlserver('BD_Medic_AI')
                with engine.connect() as conn:


                    # Obtener el IDUsuario
                    query = text("SELECT IDUsuario FROM Usuarios WHERE email = :email")
                    id_result = conn.execute(query, {"email": email}).fetchone()
                    id_usuario = id_result[0]

                    query = text("""
                                INSERT INTO Prompts (IDUsuario, Contenido, Fecha) 
                                VALUES (:id_usuario, :contenido, GETDATE())
                            """)

                    conn.execute(query, {
                        "id_usuario": id_usuario,
                        "contenido": sintomas
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
                    return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404

                edad, sexo, enfermedades, habitos = usuario

                prompt_completo = f"""Rol/Entorno: "Actúa como un médico general especializado en diagnóstico inicial de síntomas comunes."
                                        Tarea: "Analiza los siguientes síntomas del usuario: [{sintomas}]."
                                        Resultado deseado: "Devuelve el diagnóstico en formato de texto estructurado con las secciones: diagnóstico, gravedad, explicación y recomendaciones médicas."
                                        Parámetros: "Usa lenguaje claro y comprensible para un paciente no especializado, sin mencionar medicamentos ni tratamientos."
                                        Receptor: "Paciente adulto con conocimiento básico en salud. Información relevante del usuario (edad: {edad}, sexo: {sexo}, enfermedades crónicas: {enfermedades}, hábitos: {habitos}) registrada en la base de datos."
                                        """

                # COHERE
                chat_response_cohere = co.chat(
                    model=model_cohere,
                    message=prompt_completo,
                    chat_history=cohere_chat_history
                )

                cohere_chat_history.append({"role": "USER", "message": prompt_completo})
                cohere_chat_history.append({"role": "CHATBOT", "message": chat_response_cohere.text})
                # Fin COHERE

                # GEMINI
                chat_response_gemini = gemini_chat_history.send_message(prompt_completo)
                # Fin GEMINI

                # QWEN
                qwen_chat_history.append({
                    "role": "user",
                    "content": prompt_completo
                })

                response = client.chat_completion(
                    model=model_qwen,
                    messages=qwen_chat_history,
                    max_tokens=500
                )

                chat_response_qwen = response.choices[0].message.content

                qwen_chat_history.append({
                    "role": "assistant",
                    "content": chat_response_qwen
                })
                # Fin QWEN

                # MISTRAL
                mistral_chat_history.append({
                    "role": "user",
                    "content": prompt_completo
                })

                response = client.chat_completion(
                    model=model_mistral,
                    messages=mistral_chat_history,
                    max_tokens=500
                )

                chat_response_mistral = response.choices[0].message.content

                mistral_chat_history.append({
                    "role": "assistant",
                    "content": chat_response_mistral
                })
                # Fin MISTRAL

                # LLAMA
                llama_chat_history.append({
                    "role": "user",
                    "content": prompt_completo
                })

                response = client.chat_completion(
                    model=model_llama,
                    messages=llama_chat_history,
                    max_tokens=500
                )

                chat_response_llama = response.choices[0].message.content

                llama_chat_history.append({
                    "role": "assistant",
                    "content": chat_response_llama
                })
                # Fin LLAMA

                respuestas_ias = {1: chat_response_cohere.text, 2: chat_response_gemini.text, 3: chat_response_qwen,
                                  4: chat_response_mistral, 5: chat_response_llama}
                votos = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

                # Modelo Meta-Votacion / 1:COHERE / 2:GEMINI / 3:QWEN / 4:MISTRAL / 5:LLAMA /
                prompt_votacion = f"""Rol/Entorno: "Actúa como un médico general el cual va elegir entre multiples diagnosticos el mejor."
                                                Tarea: "Analiza los siguientes diagnosticos numerados: 1:[{chat_response_cohere.text}], 2:[{chat_response_gemini.text}], 3:[{chat_response_qwen}], 4:[{chat_response_mistral}], 5:[{chat_response_llama}]."
                                                Resultado deseado: "Devuelve unicamente el numero del diagnostico que elegiste, no agregues contexto ni informacion adicional, unicamente el numero del diagnostico elegido."
                                                """

                print("=" * 50)
                print("Modelo Meta-Votacion / 1:COHERE / 2:GEMINI / 3:QWEN / 4:MISTRAL / 5:LLAMA /")

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
                    print(f"⚠️ GEMINI voto inválido: {votacion_gemini.text}")                # Fin GEMINI

                # QWEN
                qwen_chat_history.append({
                    "role": "user",
                    "content": prompt_votacion
                })

                response = client.chat_completion(
                    model=model_qwen,
                    messages=qwen_chat_history,
                    max_tokens=500
                )

                votacion_qwen = response.choices[0].message.content

                qwen_chat_history.append({
                    "role": "assistant",
                    "content": votacion_qwen
                })

                print("QWEN Votacion: ", votacion_qwen)
                voto_qwen = extraer_numero_voto(votacion_qwen)
                if voto_qwen:
                    votos[voto_qwen] += 1
                else:
                    print(f"⚠️ QWEN voto inválido: {votacion_qwen}")                # Fin QWEN

                # MISTRAL
                mistral_chat_history.append({
                    "role": "user",
                    "content": prompt_votacion
                })

                response = client.chat_completion(
                    model=model_mistral,
                    messages=mistral_chat_history,
                    max_tokens=500
                )

                votacion_mistral = response.choices[0].message.content

                mistral_chat_history.append({
                    "role": "assistant",
                    "content": votacion_mistral
                })

                print("MISTRAL Votacion: ", votacion_mistral)
                voto_mistral = extraer_numero_voto(votacion_mistral)
                if voto_mistral:
                    votos[voto_mistral] += 1
                else:
                    print(f"⚠️ MISTRAL voto inválido: {votacion_mistral}")                # Fin MISTRAL

                # LLAMA
                llama_chat_history.append({
                    "role": "user",
                    "content": prompt_votacion
                })

                response = client.chat_completion(
                    model=model_llama,
                    messages=llama_chat_history,
                    max_tokens=500
                )

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
                    print(f"⚠️ LLAMA voto inválido: {votacion_llama}")                # Fin LLAMA

                print("=" * 50)

                print(votos)
                top_votos = max(votos, key=votos.get)

                prompt_seleccionado = respuestas_ias[top_votos]

                logger.info(f"Prompt completo generado para {email}")

                return jsonify({'success': True, 'respuesta': prompt_seleccionado}), 200

            else:
                # No hay coincidencias con enfermedades respiratorias
                return jsonify({
                    'success': True,
                    'prompt_invalido': True,
                    'respuesta': "Lo siento, solo puedo ayudar con síntomas relacionados a enfermedades respiratorias."
                }), 200
        else:
            return jsonify({'success': True,
                            'respuesta': "Has alcanzado el numero maximo de consultaas hoy. Intentalo nuevamente mañana"}), 200

    except Exception as e:
        logger.error(f"Error en send_prompt: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)