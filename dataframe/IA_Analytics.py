from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import matplotlib.pyplot
import pandas as pd
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
TIMEOUT_LIMIT = 25
VOTING_LIMIT = 6

''' ------------------------------------ '''

def crear_engine_sqlserver(BD):
    """Crear conexión a SQL Server"""
    server = r'.\SQLEXPRESS'
    database = BD
    connection_url = f'mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
    engine = create_engine(connection_url)
    return engine

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

def extraer_numero_voto(texto):
    """Extrae el primer número encontrado en el texto"""
    import re
    numeros = re.findall(r'\d+', str(texto))
    if numeros:
        num = int(numeros[0])
        if 1 <= num <= 5:
            return num
    return None

class RespuestaError:
    def __init__(self, text):
        self.text = text

if __name__ == '__main__':

    cronometro = Cronometro()
    tie_time = {}
    TIMEOUTS = 0

    df = pd.read_excel('BD_IRA.xlsx')
    # Limpiar nombres de columnas (elimina espacios al inicio y final)
    df.columns = df.columns.str.strip()

    for i in range(102, 802):
        registro = df.iloc[i]

        oxygen = registro['Oxygen saturation (SaO2) at admission']
        resp_rate = registro['Respiratory rate']
        temp = registro['Axillary temperature (°C)']
        heart_rate = registro['Heart rate']
        cyanosis = registro['Cyanosis']
        nasal_flaring = registro['Nasal flaring']
        laryngeal = registro['Laryngeal stridor']
        crackles = registro['Crackles']
        wheezing = registro['Wheezing']
        rhonchi = registro['Rhonchi']
        hypoventilation = registro['Hypoventilation']
        age = registro['Age (months)']
        duration_pain = registro['Duration of pain  before consultation (days)']
        cough = registro['History of cough']
        fever = registro['History of fever']
        days_fever = registro['Number of days with fever']
        asthmatic = registro['Known asthmatic patient']
        chronic_condition = registro['Patient with a diagnosed chronic condition']
        health_history = registro['Health history : Prior admission because of respiratory condition']
        sleepiness = registro['Unusual sleepiness']
        paleness = registro['Paleness']
        disorder_consciousness = registro['Disorders of consciousness']
        dehydration = registro['Dehydration signs']
        restlessness = registro['Restlessness']
        vomiting = registro['History of vomiting']
        diarrhea = registro['History of diarrhea']
        rhinorrhea = registro['History of rhinorrhea']
        weight = registro['Weight (Kg)']
        height = registro['Height (cm)']
        c_protein = registro['C-reactive protein']
        procalcitonin = registro['Procalcitonin']
        allele1 = registro['Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel)-Allele 1']
        allele2 = registro['Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel)-Allele 2']
        allele3 = registro['Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel)-Allele 3']
        allele4 = registro['Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel)-Allele 4']
        allele5 = registro['Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel)-Allele 5']
        npa1 = registro['Nasopharyngeal aspirate culture (NPA)-Identification 1']
        npa2 = registro['Nasopharyngeal aspirate culture (NPA)-Identification 2']
        x_ray = registro['Chest X-ray finding']
        pleural = registro['Pleural effusion']
        pleural_location = registro['Location of pleural effusion']

        prompt_completo = f""" Rol/Entorno: “Actúa como un médico general con experiencia en diagnóstico inicial de enfermedades respiratorias agudas”, 
                                Tarea: “Analiza la siguiente información de paciente: [Oxygen saturation (SaO2) at admission: {oxygen}, Respiratory rate: {resp_rate}, 
                                                                                        Axillary temperature (°C): {temp}, Heart rate: {heart_rate}, 
                                                                                        Cyanosis: {cyanosis}, Nasal flaring: {nasal_flaring}, 
                                                                                        Laryngeal stridor: {laryngeal}, Crackles: {crackles}, 
                                                                                        Wheezing: {wheezing}, Rhonchi: {rhonchi}, 
                                                                                        Hypoventilation: {hypoventilation}, Age (months): {age}, 
                                                                                        Duration of pain before consultation (days): {duration_pain}, History of cough: {cough}, 
                                                                                        History of fever: {fever}, Number of days with fever: {days_fever}, 
                                                                                        Known asthmatic patient: {asthmatic}, Patient with a diagnosed chronic condition: {chronic_condition}, 
                                                                                        Health history: Prior admission because of respiratory condition: {health_history}, 
                                                                                        Unusual sleepiness: {sleepiness}, Paleness: {paleness}, Disorders of consciousness: {disorder_consciousness}, 
                                                                                        Dehydration signs: {dehydration}, Restlessness: {restlessness}, History of vomiting: {vomiting}, 
                                                                                        History of diarrhea: {diarrhea}, History of rhinorrhea: {rhinorrhea}, Weight (Kg): {weight}, 
                                                                                        Height (cm): {height}, C-reactive protein: {c_protein}, Procalcitonin: {procalcitonin},
                                                                                        Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel) - Allele 1: {allele1},
                                                                                        Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel) - Allele 2: {allele2},
                                                                                        Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel) - Allele 3: {allele3},
                                                                                        Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel) - Allele 4: {allele4},
                                                                                        Detection of DNA/RNA (TrueScience Respifinder Pathogen Identification Panel) - Allele 5: {allele5},
                                                                                        Nasopharyngeal aspirate culture (NPA) - Identification 1: {npa1}, 
                                                                                        Nasopharyngeal aspirate culture (NPA) - Identification 2: {npa2}, 
                                                                                        Chest X-ray finding: {x_ray}, Pleural effusion: {pleural}, Location of pleural effusion: {pleural_location}].
                                Resultado deseado: “Devuelve la enfermedad detectada con su nombre en ingles.”, Parámetros: “Responde solo con el nombre de la enfermedad, sin texto adicional ni contexto.”"""

        print("/" * 50)
        print("NO. PATIENT: ", i + 1)

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
            #insertar_respuesta(1, chat_response_cohere.text, cohere_tiempo)

        except TimeoutError:
            cronometro.detener('cohere_time')
            print(f"""⚠️ ERROR: Cohere excedió el límite de {TIMEOUT_LIMIT} segundos""")
            tie_time[1] = None
            #insertar_respuesta(1, "TIMEOUT", None)
            chat_response_cohere = RespuestaError("TIMEOUT")
            TIMEOUTS += 1

        except Exception as e:
            cronometro.detener('cohere_time')
            print(f"⚠️ ERROR Cohere: {str(e)}")
            tie_time[1] = None
            #insertar_respuesta(1, f"ERROR: {str(e)}", None)
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
            #insertar_respuesta(2, chat_response_gemini.text, gemini_tiempo)

        except TimeoutError:
            cronometro.detener('gemini_time')
            print(f"""⚠️ ERROR: Gemini excedió el límite de {TIMEOUT_LIMIT} segundos""")
            tie_time[2] = None
            #insertar_respuesta(2, "TIMEOUT", None)
            chat_response_gemini = RespuestaError("TIMEOUT")
            TIMEOUTS += 1


        except Exception as e:
            cronometro.detener('gemini_time')
            print(f"⚠️ ERROR Gemini: {str(e)}")
            tie_time[2] = None
            #insertar_respuesta(2, f"ERROR: {str(e)}", None)
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
            #insertar_respuesta(3, chat_response_qwen, qwen_tiempo)

        except TimeoutError:
            cronometro.detener('qwen_time')
            print(f"""⚠️ ERROR: Qwen excedió el límite de {TIMEOUT_LIMIT} segundos""")
            tie_time[3] = None
            #insertar_respuesta(3, "TIMEOUT", None)
            chat_response_qwen = RespuestaError("TIMEOUT")
            TIMEOUTS += 1


        except Exception as e:
            cronometro.detener('qwen_time')
            print(f"❌ ERROR Qwen: {str(e)}")
            tie_time[3] = None
            #insertar_respuesta(3, f"ERROR: {str(e)}", None)
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
            #insertar_respuesta(4, chat_response_mistral, mistral_tiempo)

        except TimeoutError:
            cronometro.detener('mistral_time')
            print(f"""⚠️ ERROR: Mistral excedió el límite de {TIMEOUT_LIMIT} segundos""")
            tie_time[4] = None
            #insertar_respuesta(4, "TIMEOUT", None)
            chat_response_mistral = RespuestaError("TIMEOUT")

            TIMEOUTS += 1


        except Exception as e:
            cronometro.detener('mistral_time')
            print(f"❌ ERROR Mistral: {str(e)}")
            tie_time[4] = None
            #insertar_respuesta(4, f"ERROR: {str(e)}", None)
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
            #insertar_respuesta(5, chat_response_llama, llama_tiempo)

        except TimeoutError:
            cronometro.detener('llama_time')
            print(f"""⚠️ ERROR: Llama excedió el límite de {TIMEOUT_LIMIT} segundos""")
            tie_time[5] = None
            #insertar_respuesta(5, "TIMEOUT", None)
            chat_response_llama = RespuestaError("TIMEOUT")
            TIMEOUTS += 1


        except Exception as e:
            cronometro.detener('llama_time')
            print(f"❌ ERROR Llama: {str(e)}")
            tie_time[5] = None
            #insertar_respuesta(5, f"ERROR: {str(e)}", None)
            chat_response_llama = RespuestaError("ERROR")
            TIMEOUTS += 1

        respuestas_ias = {1: chat_response_cohere.text, 2: chat_response_gemini.text, 3: chat_response_qwen,
                          4: chat_response_mistral, 5: chat_response_llama}
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


        # MISTRAL
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

        max_valor = max(votos.values())
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

        max_length = 499

        engine = crear_engine_sqlserver('BD_MedicAI')
        with engine.connect() as conn:
            query = text("""
                        INSERT INTO Diagnosticos_IAs (NumeroPaciente, Sintomas, EnfermedadIA1, EnfermedadIA2, EnfermedadIA3, EnfermedadIA4, EnfermedadIA5, Enfermedad_Seleccionada, IA_Seleccionada, IRA_Diagnosticada) 
                        VALUES (:numPac, :sintomas, :ia1, :ia2, :ia3, :ia4, :ia5, :enfermedad_selec, :ia_selec, :ira_diag)
                    """)

            conn.execute(query, {
                "numPac": i + 1,
                "sintomas": prompt_completo,
                "ia1": chat_response_cohere.text[:max_length],
                "ia2": chat_response_gemini.text[:max_length],
                "ia3": chat_response_qwen[:max_length],
                "ia4": chat_response_mistral[:max_length],
                "ia5": chat_response_llama[:max_length],
                "enfermedad_selec": respuestas_ias[ia_seleccionada][:max_length],
                "ia_selec": ia_seleccionada,
                "ira_diag": registro['Main diagnostic']
            })
            conn.commit()

