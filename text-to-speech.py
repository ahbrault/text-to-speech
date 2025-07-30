#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import datetime
import re
import os
import glob
from pydub.effects import speedup
from tqdm import tqdm
from pydub import AudioSegment
import nltk
from TTS.api import TTS

def normalize_text(text: str) -> str:
    """
    Nettoie et normalise le texte pour le rendre compatible avec le modèle TTS.
    """
    text = re.sub(r"[’‘]", "'", text)
    text = re.sub(r'[“”«»]', '"', text)
    text = re.sub(r"—", "-", text)
    text = re.sub(r"[\/|*~]", " ", text)
    text = text.replace("€", "euros").replace("$", "dollars").replace("%", "pourcent")
    text = text.replace("\u200c", "")
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-"
    text = ''.join(filter(lambda char: char in allowed_chars, text))
    return text

def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Text-to-Speech CLI, un outil avancé pour la synthèse vocale.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Groupes d'arguments pour une aide plus claire ---
    input_group = parser.add_argument_group('Source du texte')
    output_group = parser.add_argument_group('Sortie Audio')
    model_group = parser.add_argument_group('Sélection du Modèle et de la Voix')
    ui_group = parser.add_argument_group('Interface Utilisateur')
    info_group = parser.add_argument_group('Information')

    # --- Arguments ---
    input_group.add_argument("text", nargs="*", help="Texte à synthétiser. Ignoré si --file est utilisé.")
    input_group.add_argument("-f", "--file", type=str, help="Fichier texte à synthétiser.")

    output_group.add_argument("-o", "--output", type=str, help="Nom du fichier de sortie .wav (sans extension).")
    output_group.add_argument("--speed", type=float, default=1.0, help="Vitesse de la parole (ex: 1.0 normal, 1.2 plus rapide, 0.9 plus lent).")
    
    model_group.add_argument("--fr", action="store_true", help="Utilise la voix française par défaut.")
    model_group.add_argument("--en", action="store_true", help="Utilise la voix anglaise par défaut (comportement par défaut).")
    model_group.add_argument("--model", type=str, help="Nom exact du modèle TTS à utiliser (ex: 'tts_models/en/vctk/vits'). Prioritaire sur --fr/--en.")
    model_group.add_argument("--speaker", type=str, help="ID du locuteur à utiliser pour les modèles multi-locuteurs.")

    ui_group.add_argument("-q", "--quiet", action="store_true", help="Mode silencieux. N'affiche pas la barre de progression ni les infos.")

    info_group.add_argument("--list-models", action="store_true", help="Affiche la liste de tous les modèles disponibles et quitte.")
    info_group.add_argument("--list-speakers", action="store_true", help="Affiche les locuteurs pour un modèle donné (doit être utilisé avec --model).")

    args = parser.parse_args()
    
    # --- Fonctions d'information et sortie ---
    if args.list_models:
        print("Models disponibles pour TTS:")
        for model in TTS.list_models():
            print(model)
        sys.exit(0)

    if args.list_speakers:
        if not args.model:
            print("❌ Erreur : L'argument --model est requis pour lister les locuteurs.")
            sys.exit(1)
        print(f"⏳ Chargement du modèle '{args.model}' pour trouver les locuteurs...")
        try:
            tts_instance = TTS(args.model)
            if tts_instance.is_multi_speaker:
                print(f"✅ Locuteurs disponibles pour '{args.model}':")
                for speaker_name in sorted(tts_instance.speakers):
                    print(f"- {speaker_name}")
            else:
                print("ℹ️ Ce modèle n'est pas multi-locuteur.")
        except Exception as e:
            print(f"❌ Impossible de charger le modèle : {e}")
        sys.exit(0)

    # --- Fonction d'affichage silencieux ---
    def print_q(*p_args, **p_kwargs):
        if not args.quiet:
            print(*p_args, **p_kwargs)

    # --- Configuration du modèle ---
    MODELS = {
        "en": {"model_name": "tts_models/en/ljspeech/tacotron2-DDC", "lang_nltk": "english"},
        "fr": {"model_name": "tts_models/fr/mai/tacotron2-DDC", "lang_nltk": "french"},
    }

    if args.model:
        model_name = args.model
        # Détection de la langue pour NLTK à partir du nom du modèle
        lang_code = model_name.split('/')[1] if '/' in model_name else 'en'
        lang_nltk = "french" if lang_code == 'fr' else 'english'
    else:
        lang_code = "fr" if args.fr else "en"
        config = MODELS[lang_code]
        model_name = config["model_name"]
        lang_nltk = config["lang_nltk"]

    # --- Récupération et normalisation du texte ---
    print_q("1️⃣  Normalisation et préparation du texte...")
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"❌ Fichier non trouvé : {args.file}"); sys.exit(1)
    elif args.text:
        full_text = " ".join(args.text)
    else:
        print("❌ Aucun texte fourni."); parser.print_help(); sys.exit(1)

    if not full_text.strip():
        print("❌ Le texte fourni est vide."); sys.exit(1)

    normalized_text = normalize_text(full_text)
    sentences = nltk.sent_tokenize(normalized_text, language=lang_nltk)
    
    if not sentences:
        print("❌ Après nettoyage, aucun texte à synthétiser n'a été trouvé."); sys.exit(1)

    # --- Initialisation de TTS ---
    print_q(f"2️⃣  Initialisation du modèle TTS ({model_name})...")
    tts = TTS(model_name=model_name, progress_bar=False)

    # Vérification du locuteur
    speaker_to_use = None
    if args.speaker:
        if tts.is_multi_speaker:
            if args.speaker in tts.speakers:
                speaker_to_use = args.speaker
            else:
                print_q(f"⚠️ Locuteur '{args.speaker}' non valide. Utilisation du locuteur par défaut.")
        else:
            print_q("⚠️ L'argument --speaker a été fourni, mais le modèle n'est pas multi-locuteur.")

    # --- Synthèse vocale ---
    print_q(f"3️⃣  Génération de l'audio pour {len(sentences)} phrases...")
    temp_dir = "temp_audio_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    chunk_files = []
    
    for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Synthèse", disable=args.quiet):
        sentence = sentence.strip()
        if not sentence: continue

        chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
        try:
            tts.tts_to_file(text=sentence, file_path=chunk_file, speaker=speaker_to_use)
            chunk_files.append(chunk_file)
        except Exception as e:
            print_q(f"\n⚠️ Erreur lors de la synthèse de la phrase {i+1}. Elle sera ignorée. Erreur: {e}")

    # --- Assemblage et post-traitement ---
    if not chunk_files:
        print("❌ Aucune phrase n'a pu être synthétisée."); sys.exit(1)

    print_q("4️⃣  Assemblage et post-traitement audio...")
    combined_audio = AudioSegment.empty()
    for chunk_file in sorted(chunk_files):
        try:
            sound = AudioSegment.from_wav(chunk_file)
            combined_audio += sound
        except Exception:
            print_q(f"\n⚠️ Impossible de lire le chunk {chunk_file}. Il sera ignoré.")

    # Application de la vitesse
    if args.speed != 1.0:
        print_q(f"      - Application de la vitesse x{args.speed}...")
        combined_audio = speedup(combined_audio, playback_speed=args.speed)

    # --- Exportation et nettoyage ---
    if args.output:
        output_file = f"{args.output}.wav"
    else:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output-{lang_code}_{now}.wav"

    print_q(f"5️⃣  Exportation du fichier final : {output_file}")
    combined_audio.export(output_file, format="wav")

    print_q("6️⃣  Nettoyage des fichiers temporaires...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    os.rmdir(temp_dir)

    print(f"✅ Terminé ! Fichier généré : {output_file}")

if __name__ == '__main__':
    # Vérification des dépendances NLTK au début
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except nltk.downloader.DownloadError:
        print("⏳ Téléchargement des paquets de données NLTK nécessaires...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ Paquets de données téléchargés.")
    main()
