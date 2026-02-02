🛡️ AI Multi-Model Spam Detector
¿De qué va este proyecto?
Este proyecto es un sistema inteligente de clasificación de correo electrónico que utiliza tres modelos distintos de Machine Learning (MultinomialNB, Decision Tree y GaussianNB) para identificar mensajes no deseados. Mediante el uso de Procesamiento de Lenguaje Natural (NLP), la aplicación limpia, lematiza y analiza el texto para determinar en tiempo real si un mensaje es Spam o Seguro (Ham).

📥 Instalación y Descarga
El proyecto utiliza Git LFS (Large File Storage) para gestionar los archivos de datos y los modelos entrenados (.csv y .pkl), asegurando que el repositorio se mantenga ligero y funcional.

1. Descargar Git LFS
Si no lo tienes en tu sistema, descárgalo e instálalo desde la página oficial: 👉 git-lfs.com

2. Clonar el repositorio
Ejecuta los siguientes comandos en tu terminal para bajar el código y los archivos pesados:

Bash
# Clonar el repositorio
git clone https://github.com/jabbvk/spam-detector-ia.git

# Entrar en la carpeta
cd spam-detector-ia

# Inicializar y descargar los archivos LFS
git lfs install
git lfs pull
⚙️ Configuración y Ejecución
Sigue estos pasos para preparar tu entorno y lanzar la interfaz gráfica:

1. Instalar dependencias
Asegúrate de tener Python instalado y ejecuta el siguiente comando para instalar las librerías necesarias:

Bash
pip install scikit-learn pandas joblib matplotlib seaborn Pillow nltk
2. Lanzar la aplicación
Una vez instaladas las dependencias, simplemente ejecuta el archivo principal de la interfaz:

Bash
python main_app.py
🚀 Cómo usarlo
Al abrir la aplicación, el sistema verificará automáticamente si existen los modelos entrenados. Si falta alguno, la terminal integrada te informará mientras realiza el entrenamiento automáticamente.

Escribe o pega el texto de un email en el cuadro superior.

Pulsa "EJECUTAR ANÁLISIS IA".

Verás la opinión de los tres modelos simultáneamente y podrás consultar sus Matrices de Confusión pulsando en los botones correspondientes.