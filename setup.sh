#!/bin/bash
set -e # Arrête le script si une commande échoue

echo "--- Configuration de l'outil Text-to-Speech ---"

# 1. Vérification de la présence de Python 3
echo "1/5 : Vérification de Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "ERREUR : Python 3 n'est pas installé. Veuillez l'installer pour continuer."
    exit 1
fi
echo "      Python 3 trouvé."

# 2. Création de l'environnement virtuel
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "2/5 : Création de l'environnement virtuel dans '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
else
    echo "2/5 : Environnement virtuel existant trouvé."
fi


# 3. Activation et installation des dépendances
echo "3/5 : Activation et installation des dépendances (cela может prendre plusieurs minutes)..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip > /dev/null
pip install -r requirements.txt

# 4. Téléchargement des données NLTK
echo "4/5 : Téléchargement des paquets de données NLTK..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 5. Rendre les scripts exécutables
echo "5/5 : Finalisation des permissions..."
chmod +x text-to-speech.py
chmod +x tts

echo ""
echo "✅ Installation terminée avec succès !"
echo ""
echo "--- Comment utiliser ---"
echo "Pour lancer le programme, utilisez la commande './tts' depuis ce dossier."
echo "Exemple : ./tts -f votre_fichier.txt --fr -o sortie_audio"
echo "Pour voir toutes les options : ./tts --help"
