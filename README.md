# 🛡️ AI Multi-Model Spam Detector

### **¿De qué va este proyecto?**
Este proyecto consiste en un sistema inteligente de detección de correo basura (Spam) que emplea tres modelos de Machine Learning: Multinomial Naive Bayes, Árbol de Decisión y Naive Bayes Gaussiano. A través de técnicas de Procesamiento de Lenguaje Natural (NLP), la herramienta limpia, normaliza y analiza el texto para clasificar mensajes en tiempo real.

---

### **📥 Instalación y Descarga**

El repositorio utiliza **Git LFS (Large File Storage)** para gestionar el dataset y los archivos de los modelos entrenados.

#### **1. Descargar Git LFS**
Es necesario tener instalado Git LFS para bajar los archivos pesados correctamente. Descárgalo aquí:
👉 [**git-lfs.com**](https://git-lfs.com/)

#### **2. Clonar el repositorio**
Utiliza los siguientes comandos en tu terminal para obtener el código:

```bash
# Clonar el repositorio
git clone https://github.com/jabbvk/spam-detector-ia.git

# Entrar en el directorio
cd spam-detector-ia

# Inicializar y descargar los archivos LFS (.csv y .pkl)
git lfs install
git lfs pull
```

---

### **⚙️ Configuración y Ejecución**

#### **1. Instalar dependencias**
Instala las librerías necesarias de Python (Scikit-Learn, Pandas, Joblib, Matplotlib, Seaborn, Pillow y NLTK):

```bash
pip install scikit-learn pandas joblib matplotlib seaborn Pillow nltk
```

#### **2. Lanzar la aplicación**
Ejecuta el script principal para abrir la interfaz gráfica de usuario (GUI):

```bash
python main_app.py
```

---

### **🚀 Cómo usarlo**
1. **Inicialización**: Al arrancar, el sistema detectará automáticamente si los archivos `.pkl` de los modelos están presentes. Si falta alguno, la terminal integrada informará del progreso mientras realiza el entrenamiento en segundo plano.
2. **Análisis**: Escribe o pega el texto del mensaje que deseas comprobar en el cuadro de entrada principal.
3. **Resultados**: Pulsa el botón naranja **"EJECUTAR ANÁLISIS IA"**. Los tres modelos (Multinomial, Árbol y Gaussiano) darán su veredicto de forma simultánea.
4. **Métricas**: Para entender mejor por qué el modelo ha tomado esa decisión, pulsa en los botones **"VIEW MATRIX"** de cada tarjeta para abrir la imagen de la matriz de confusión correspondiente.