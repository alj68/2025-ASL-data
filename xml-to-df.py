import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os

def fix_conventions(text):
    """
    Fixes problematic gloss conventions in ASL sentences.
    """
    if text[-1] == "+":
        text = text.replace("+", "")

    if "(" in text:
        first = text.index("(")
        last = text.index(")")
        remove = text[first:last+1]
        text = text.replace(remove, "")

    if "alt." in text:
        text = text.replace("alt.", "")

    if "\"" in text:
        text = ""

    return text

def parse_utterances_from_xml(file_path):
    """
    Parses one XML file and returns a list of dicts with:
      - translation: the English sentence (without surrounding quotes)
      - gloss: the ASL gloss sentence (concatenated LABELs)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    utterances_data = []

    for utt in root.findall(".//UTTERANCE"):
        tr = utt.find("TRANSLATION")
        if tr is None or not tr.text:
            continue
        translation = tr.text.strip().strip("\'")

        labels = []
        for sign in utt.findall(".//MANUALS/SIGN"):
            lbl = sign.find("LABEL")
            if lbl is not None and lbl.text:
                lbl_text = fix_conventions(lbl.text.strip().strip("'\""))
                labels.append(lbl_text)
        gloss = " ".join(labels)

        utterances_data.append({
            "translation": translation,
            "gloss": gloss
        })

    return utterances_data

def build_dataframe_from_folder(xml_folder):
    """
    Walks through all .xml files in xml_folder and builds a DataFrame.
    """
    all_records = []
    for xml_file in glob.glob(os.path.join(xml_folder, "*.xml")):
        all_records.extend(parse_utterances_from_xml(xml_file))

    df = pd.DataFrame(all_records, columns=["translation", "gloss"])
    return df

if __name__ == "__main__":
    # ←– Point this to your folder of .xml files
    xml_folder = "Z:\\PreComputedFiles\\ASLLRP-Sentence.Gloss-Data\\extracted_xml"

    df = build_dataframe_from_folder(xml_folder)
    print(f"Found {len(df)} utterances across all files.")
    print(df.head(10))

    # Optional: save to disk
    df.to_csv("all_asl_utterances.csv", index=False)
