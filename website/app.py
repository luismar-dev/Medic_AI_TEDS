from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
import hashlib
import os

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
                INSERT INTO Usuarios (Nombre, email, edad, sexo, enfermedades, habitos, contrasena, cuenta_verificada, fecha_registro, fecha_verificacion) 
                VALUES (:nombre, :email, :edad, :sexo, :enfermedades, :habitos, :contrasena, 1, :fecha, :fecha)
            """)

            conn.execute(query, {
                "nombre": datos['nombre'],
                "email": datos['email'],
                "edad": int(datos['edad']),
                "sexo": datos['sexo'],
                "enfermedades": datos['enfermedades'],
                "habitos": datos['habitos'],
                "contrasena": contrasena_hash,
                "fecha": datetime.now()
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
                    'id': row[0] if len(row) > 0 else None,
                    'nombre': row[1] if len(row) > 1 else None,
                    'email': row[2] if len(row) > 2 else None,
                    'edad': row[3] if len(row) > 3 else None,
                    'sexo': row[4] if len(row) > 4 else None,
                    'enfermedades': row[5] if len(row) > 5 else None,
                    'habitos': row[6] if len(row) > 6 else None
                })

        return jsonify({'usuarios': usuarios}), 200

    except Exception as e:
        logger.error(f"Error al obtener usuarios: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que el servidor está corriendo"""
    return jsonify({'status': 'OK', 'message': 'Servidor funcionando correctamente'}), 200


@app.route('/api/send-prompt', methods=['POST'])
def send_prompt():
    """
    Recibe los síntomas, busca los datos del usuario en la BD,
    construye el prompt completo y lo devuelve.
    """
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

        logger.info(f"Prompt completo generado para {email}")

        return jsonify({'success': True, 'prompt_completo': prompt_completo}), 200

    except Exception as e:
        logger.error(f"Error en send_prompt: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
