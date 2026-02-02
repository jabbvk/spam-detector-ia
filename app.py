import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import os
import joblib

# Importamos tus scripts originales
import preparar_datos
import entrenar_modelo_multinomial
import entrenar_modelo_arbol
import entrenar_modelo_gaussiano

class AppSpam:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Spam - Ventana de Control")
        self.root.geometry("600x700")

        # 1. Área de Input
        tk.Label(root, text="Introduce el mensaje a analizar:", font=('Arial', 12, 'bold')).pack(pady=10)
        self.input_text = tk.Text(root, height=8, width=60)
        self.input_text.pack(pady=5)

        # 2. Botón de Comprobar
        self.btn_comprobar = tk.Button(root, text="🔍 COMPROBAR MENSAJE", command=self.predecir, 
                                       bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'), state="disabled")
        self.btn_comprobar.pack(pady=10)

        # 3. Mini Terminal de Logs
        tk.Label(root, text="Terminal de estado:", font=('Arial', 10, 'italic')).pack(pady=(20, 0))
        self.log_area = scrolledtext.ScrolledText(root, height=15, width=70, bg="black", fg="#00FF00")
        self.log_area.pack(pady=5)

        # Iniciar proceso de verificación de archivos al abrir
        self.log("Iniciando aplicación...")
        threading.Thread(target=self.inicializar_sistema, daemon=True).start()

    def log(self, mensaje):
        """Escribe en la terminal de la ventana"""
        self.log_area.insert(tk.END, f"> {mensaje}\n")
        self.log_area.see(tk.END)

    def inicializar_sistema(self):
        """Revisa pkls y los genera si no existen"""
        try:
            # 1. Preparar datos
            if not os.path.exists("datos_procesados.pkl"):
                self.log("AVISO: No se detectó 'datos_procesados.pkl'.")
                self.log("Preparando datos desde 'spam.csv'...")
                # Usamos el mismo archivo para train y test por simplicidad si no tienes dos
                exito, msg = preparar_datos.ejecutar_preparacion("spam.csv", "spam.csv", "datos_procesados.pkl")
                if not exito: 
                    self.log(f"ERROR: {msg}")
                    return
                self.log("Datos procesados correctamente.")
            
            # 2. Entrenar modelos si no existen
            modelos = [
                ("modelo_multinomial.pkl", entrenar_modelo_multinomial.entrenar),
                ("modelo_arbol.pkl", entrenar_modelo_arbol.entrenar),
                ("modelo_gaussiano.pkl", entrenar_modelo_gaussiano.entrenar)
            ]

            for nombre_pkl, funcion_entrenar in modelos:
                if not os.path.exists(nombre_pkl):
                    self.log(f"Entrenando {nombre_pkl}...")
                    funcion_entrenar("datos_procesados.pkl", nombre_pkl, f"cm_{nombre_pkl.replace('.pkl', '.png')}")
                    self.log(f"Modelo {nombre_pkl} listo.")

            self.log("SISTEMA LISTO. Ya puedes analizar mensajes.")
            self.btn_comprobar.config(state="normal")

        except Exception as e:
            self.log(f"ERROR CRÍTICO: {str(e)}")

    def predecir(self):
        texto = self.input_text.get("1.0", tk.END).strip()
        if not texto:
            messagebox.showwarning("Atención", "Escribe un mensaje primero.")
            return

        try:
            # Cargar lo necesario
            data = joblib.load("datos_procesados.pkl")
            vec = data['vectorizador']
            modelo = joblib.load("modelo_multinomial.pkl") # Usamos Multinomial por defecto

            # Limpiar y transformar usando tus funciones de preparar_datos.py
            # Replicamos la lógica de limpieza de tu script
            t = preparar_datos.tokenizar(texto)
            t = preparar_datos.hacer_minusculas(t)
            t = preparar_datos.eliminar_simbolos(t)
            t = preparar_datos.eliminar_numeros(t)
            t = preparar_datos.quitar_invariantes(t)
            t = preparar_datos.lematizar(t)
            limpio = "".join(i+' ' for i in t)

            vector = vec.transform([limpio]).toarray()
            resultado = modelo.predict(vector)[0]

            if resultado == 'spam':
                messagebox.showwarning("Resultado", "🚨 ALERTA: Es SPAM")
            else:
                messagebox.showinfo("Resultado", "✅ MENSAJE SEGURO (HAM)")

        except Exception as e:
            self.log(f"Error en predicción: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppSpam(root)
    root.mainloop()