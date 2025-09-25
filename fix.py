# Programm zur Behebung des UTF-8-Encoding-Problems

def fix_encoding(file_path):
    # Ã–ffnen der Datei zum Lesen
    with open(file_path, 'r', encoding='latin1') as file:
        lines = file.readlines()

    # ÃœberprÃ¼fen, ob die Encoding-Zeile bereits vorhanden ist
    if not lines[0].startswith("# -*- coding: utf-8 -*-"):
        # UTF-8-Encoding-Zeile hinzufÃ¼gen
        lines.insert(0, "# -*- coding: utf-8 -*-\n")

    # Schreiben der geÃ¤nderten Datei
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# Beispiel fÃ¼r die Verwendung
file_path ='serve+.py' #"musicanno.py" #   # Pfad zur Datei
fix_encoding(file_path)
