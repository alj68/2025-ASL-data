import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os

def extract_text(track):
    """
    Extracts the text content found in <A> tags within a TRACK element.
    """
    return track.text

def parse_utterances_from_xml(file_path):
    """
    Parses one XML file and returns a list of:
      - the English sentence (without surrounding quotes)
      - the ASL gloss sentence (concatenated LABELs)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    utterance_pairs = []

    for utt in root.iter("UTTERANCE"):
        eng_sent = []
        asl_gloss = []
        for track in utt.iter("TRACK"):
            if track.get("FID") == "10000":
                for a in track:
                    gloss = extract_text(a)
                    if gloss is not None:
                        asl_gloss.append(gloss)
            elif track.get("FID") == "20001":
                for a in track:
                    text = extract_text(a)
                    eng_sent.append(text)
        asl_sent = " ".join(asl_gloss)

        utterance_pairs.append({
            "translation": eng_sent[0],
            "gloss": asl_sent
        })
    return utterance_pairs

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
    xml_folder = "Z:\\official-code\\data\\trial_xml"

    df = build_dataframe_from_folder(xml_folder)
    print(f"Found {len(df)} utterances across all files.")
    print(df.head(10))

    # Optional: save to disk
    df.to_csv("all_asl_utterances.csv", index=False)
