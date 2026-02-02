import tkinter as tk
from tkinter import scrolledtext, messagebox, Toplevel
import threading
import os
import joblib
from PIL import Image, ImageTk

# Importamos tus módulos
import prepare_data
import train_multinomial_model
import train_tree_model
import train_gaussian_model

# --- PALETA DARK INDUSTRIAL ---
COLOR_BG = "#21252b"          # Fondo principal (Gris muy oscuro)
COLOR_PANEL = "#2c313a"       # Fondo de las tarjetas de modelos
COLOR_INPUT = "#3b4048"       # Fondo del cuadro de texto
COLOR_TEXT_DIM = "#abb2bf"    # Texto secundario (Gris claro)
COLOR_TEXT_BRIGHT = "#ffffff" # Texto principal (Blanco)
COLOR_ACCENT = "#ff9800"      # NARANJA (Botón y títulos)
COLOR_BTN_TEXT = "#000000"    # NEGRO (Texto botón)
COLOR_TERM_BG = "#181a1f"     # Negro profundo para terminal

class SpamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Spam Detector - Dark Mode")
        self.root.geometry("1000x720")
        self.root.configure(bg=COLOR_BG)

        # --- CABECERA ---
        tk.Label(root, text="🛡️ AI SPAM DETECTOR", font=('Impact', 24), 
                 bg=COLOR_BG, fg=COLOR_ACCENT).pack(pady=20)

        # --- PANEL DE ENTRADA ---
        input_frame = tk.Frame(root, bg=COLOR_BG)
        input_frame.pack(fill="x", padx=50)
        
        tk.Label(input_frame, text="MENSAJE A ANALIZAR:", font=('Segoe UI', 10, 'bold'), 
                 bg=COLOR_BG, fg=COLOR_TEXT_DIM).pack(anchor="w")
        
        self.input_text = tk.Text(input_frame, height=4, font=('Consolas', 11), 
                                  relief="flat", bg=COLOR_INPUT, fg=COLOR_TEXT_BRIGHT,
                                  insertbackground="white", padx=10, pady=10)
        self.input_text.pack(fill="x", pady=5)

        # BOTÓN NARANJA Y NEGRO
        self.check_button = tk.Button(
            input_frame, text="EJECUTAR ANÁLISIS IA", command=self.predict_all,
            bg=COLOR_ACCENT, fg=COLOR_BTN_TEXT, font=('Segoe UI', 12, 'bold'),
            activebackground="#e68a00", activeforeground=COLOR_BTN_TEXT,
            relief="flat", cursor="hand2", state="disabled", pady=12
        )
        self.check_button.pack(fill="x", pady=15)

        # --- PANEL DE MODELOS (Gris medio, no blanco) ---
        self.results_frame = tk.Frame(root, bg=COLOR_BG)
        self.results_frame.pack(fill="x", padx=20, pady=10)
        
        self.model_widgets = {}
        models = [
            ("Multinomial", "multinomial_model.pkl", "cm_multinomial.png", train_multinomial_model.train),
            ("Decision Tree", "tree_model.pkl", "cm_tree.png", train_tree_model.train),
            ("Gaussian", "gaussian_model.pkl", "cm_gaussian.png", train_gaussian_model.train)
        ]

        for i, (name, pkl, img, func) in enumerate(models):
            # Tarjetas oscuras
            frame = tk.Frame(self.results_frame, bg=COLOR_PANEL, padx=15, pady=15, 
                             highlightbackground="#3e4451", highlightthickness=1)
            frame.grid(row=0, column=i, padx=10, sticky="nsew")
            self.results_frame.columnconfigure(i, weight=1)

            tk.Label(frame, text=name.upper(), font=('Segoe UI', 9, 'bold'), 
                     bg=COLOR_PANEL, fg=COLOR_TEXT_DIM).pack()

            res_label = tk.Label(frame, text="---", font=('Segoe UI', 14, 'bold'), 
                                 bg=COLOR_PANEL, fg=COLOR_TEXT_DIM)
            res_label.pack(pady=10)
            
            # Botón de matriz integrado
            btn_matrix = tk.Button(frame, text="Ver matriz", font=('Segoe UI', 7, 'bold'),
                                   bg="#4b5263", fg=COLOR_BTN_TEXT, relief="flat",
                                   activebackground=COLOR_ACCENT, cursor="hand2",
                                   command=lambda p=img: self.show_matrix_window(p))
            btn_matrix.pack(pady=5)

            self.model_widgets[name] = {"res": res_label, "pkl": pkl, "func": func}

        # --- TERMINAL DE LOGS ---
        log_label = tk.Label(root, text="SISTEMA DE LOGS:", font=('Segoe UI', 8, 'bold'), 
                             bg=COLOR_BG, fg=COLOR_TEXT_DIM)
        log_label.pack(padx=50, anchor="w", pady=(20,0))
        
        self.log_area = scrolledtext.ScrolledText(root, height=10, bg=COLOR_TERM_BG, 
                                                  fg=COLOR_ACCENT, font=('Consolas', 10),
                                                  borderwidth=0, padx=10, pady=10)
        self.log_area.pack(fill="x", padx=50, pady=10)

        threading.Thread(target=self.initialize_system, daemon=True).start()

    def log(self, message):
        self.log_area.insert(tk.END, f"> {message}\n")
        self.log_area.see(tk.END)

    def initialize_system(self):
        try:
            if not os.path.exists("processed_data.pkl"):
                self.log("Preparando datos procesados...")
                success, _ = prepare_data.run_preparation("spam.csv", "spam.csv", "processed_data.pkl")
                if not success: return

            for name, data in self.model_widgets.items():
                pkl_path = data["pkl"]
                if not os.path.exists(pkl_path):
                    self.log(f"Entrenando {name}...")
                    data["func"]("processed_data.pkl", pkl_path, f"cm_{name.lower().replace(' ', '')}.png")
                self.log(f"Modelo {name} cargado.")

            self.log("SISTEMA ONLINE. Esperando entrada...")
            self.check_button.config(state="normal")
        except Exception as e:
            self.log(f"ERROR: {e}")

    def show_matrix_window(self, img_path):
        if not os.path.exists(img_path):
            messagebox.showinfo("Info", "Entrenando...")
            return
        top = Toplevel()
        top.title("Confusion Matrix")
        top.configure(bg=COLOR_BG)
        img = Image.open(img_path).resize((450, 450), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image=photo, bg=COLOR_BG)
        lbl.image = photo
        lbl.pack(padx=20, pady=20)

    def predict_all(self):
        text = self.input_text.get("1.0", tk.END).strip()
        if not text: return

        try:
            data = joblib.load("processed_data.pkl")
            vec = data['vectorizador']
            
            # Limpieza profesional
            tokens = prepare_data.tokenize(text)
            tokens = prepare_data.to_lowercase(tokens)
            tokens = prepare_data.remove_symbols(tokens)
            tokens = prepare_data.remove_stopwords(tokens)
            tokens = prepare_data.lemmatize(tokens)
            clean_text = " ".join(tokens)
            
            vec_input = vec.transform([clean_text]).toarray()

            for name, widgets in self.model_widgets.items():
                model = joblib.load(widgets["pkl"])
                pred = model.predict(vec_input)[0]
                # Colores neón para resultados: Rojo vivo vs Verde neón
                color = "#ff5555" if pred == 'spam' else "#50fa7b"
                widgets["res"].config(text=pred.upper(), fg=color)
            
            self.log("Análisis finalizado.")
        except Exception as e:
            self.log(f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamApp(root)
    root.mainloop()