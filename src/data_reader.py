import spacy
from nltk import sent_tokenize
from pathlib import Path
from torch.nn import Module
from transformers import RobertaTokenizer
from typing import *

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')
nlp = spacy.load("en_core_web_sm")

def get_relation_id(rel_type: str):
    rel_id_dict = {"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
    return rel_id_dict[rel_type]


def token_id_lookup(token_span_SENT: List[List[int]], start_char: int, end_char: int):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index


def sntc_id_lookup(data_dict: Dict[str, Any], start_char: int, end_char: Optional[int] = None) -> int:
    for sntc_dict in data_dict['sentences']:
        if end_char is None:
            if start_char >= sntc_dict['sent_start_char'] and start_char <= sntc_dict['sent_end_char']:
                return sntc_dict['sent_id']
        else:
            if start_char >= sntc_dict['sent_start_char'] and end_char <= sntc_dict['sent_end_char']:
                return sntc_dict['sent_id']


def id_lookup(span_SENT: List[List[int]], start_char: int) -> int:
    """
    this function is applicable to RoBERTa subword or token from ltf/spaCy
    id: start from 0
    """
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found.")
    return token_id


def tokenized_to_origin_span(doc_content: str, tokenized_sntc: List[str]) -> List[List[int]]:
    """
    get start and end position of each sentence in document content
    """
    token_span_list = []
    pointer = 0
    for sntc in tokenized_sntc:
        while True:
            if sntc[0] == doc_content[pointer]:
                start = pointer
                end = start + len(sntc) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span_list.append([start, end])
    return token_span_list


def span_SENT_to_DOC(token_span_SENT: List[List[int]], sent_start: int):
    """
    sent_start: start position of sentence
    """
    token_span_DOC = []
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        token_span_DOC.append([start_char, end_char])
    return token_span_DOC


def RoBERTa_list(content: str, token_span_SENT: List[List[int]]):
    encoded = tokenizer.encode(content) # list of integers
    roberta_subword_to_ID = encoded

    roberta_subwords = []
    roberta_subwords_no_space = []

    for id in encoded:
        decoded_token = tokenizer.decode([id])
        roberta_subwords.append(decoded_token)
        if decoded_token[0] == " ":
            roberta_subwords_no_space.append(decoded_token[1:])
        else:
            roberta_subwords_no_space.append(decoded_token)

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1: -1]) # w/o <s> and </s>
    roberta_subword_map = []
    assert token_span_SENT is not None

    # roberta_subword_map stores each subword's id
    roberta_subword_map.append(-1) # "<s>" => -1
    for subword in roberta_subword_span:
        start_char, end_char = subword[0], subword[1]
        roberta_subword_map.append(token_id_lookup(token_span_SENT, start_char, end_char))
    roberta_subword_map.append(-1)  # "</s>" => -1

    return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map


def RoBERTa_tokenize(sntc_dict: Dict[str, Any]) -> Tuple[Union[Any]]:
    return RoBERTa_list(sntc_dict["content"], sntc_dict["token_span_SENT"])


def document_to_sentences(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    doc_content = data_dict["doc_content"]

    tokenized_sntc = sent_tokenize(doc_content) # split document into sentences
    sntc_span = tokenized_to_origin_span(doc_content, tokenized_sntc)

    for i, sntc in enumerate(tokenized_sntc):
        sntc_dict = {}
        sntc_dict["sent_id"] = i
        sntc_dict["content"] = sntc
        sntc_dict["sent_start_char"] = sntc_span[i][0]
        sntc_dict["sent_end_char"] = sntc_span[i][1]

        spacy_tokens = nlp(sntc_dict["content"])
        sntc_dict["tokens"] = []
        sntc_dict["pos"] = []

        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_tokens:
            sntc_dict["tokens"].append(token.text)
            sntc_dict["pos"].append(token.pos_)
        sntc_dict["token_span_SENT"] = tokenized_to_origin_span(sntc, sntc_dict["tokens"]) # get start and end position in sentence
        sntc_dict["token_span_DOC"] = span_SENT_to_DOC(sntc_dict["token_span_SENT"], sntc_dict["sent_start_char"]) # get start and end position in document

        roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map = RoBERTa_tokenize(sntc_dict)
        sntc_dict["roberta_subword_to_ID"] = roberta_subword_to_ID
        sntc_dict["roberta_subwords"] = roberta_subwords
        sntc_dict["roberta_subword_span_SENT"] = roberta_subword_span
        sntc_dict["roberta_subword_map"] = roberta_subword_map

        sntc_dict["roberta_subword_span_DOC"] = span_SENT_to_DOC(sntc_dict["roberta_subword_span_SENT"],
                                                                 sntc_dict["sent_start_char"])

        # store roberta POS tag for each subword
        sntc_dict["roberta_subword_pos"] = []
        for token_id in sntc_dict["roberta_subword_map"]:
            if token_id is None or token_id == -1:
                sntc_dict["roberta_subword_pos"].append("None")
            else:
                sntc_dict["roberta_subword_pos"].append(sntc_dict["pos"][token_id])

        data_dict["sentences"].append(sntc_dict)
    return data_dict


def read_tsvx_file(data_dir: Union[Path, str], file: str) -> Dict[str, Any]:
    """
    tsvx split delimeter is \t
    Text \t doc_content
    Event \t event_id \t event_trigger(word) \t event_type[Occurrence/I_Action/..] \t char start location
    Relation \t event_id1 \t event_id2 \t relation_type \t T/F \t event_word1 \t event_word2
    (event_id1 -> event_word1, event_id2 -> event_word2)

    tsvx format example:
    Text    Syrian rebels delivered six U.N.
    Event   1   delivered   Occurence   14
    Relation    20  21  NoRel	true	wished	drew
    """
    data_dict = {}
    data_dict["doc_id"] = file.replace(".tsvx", "")
    data_dict["event_dict"] = {}
    data_dict["sentences"] = []
    data_dict["relation_dict"] = {}

    file_path = data_dir / file
    for line in open(file_path, mode="r"):
        line = line.split("\t")
        type = line[0].lower()
        if type == "text":
            data_dict["doc_content"] = line[1]
        elif type == "event":
            event_id, event_word, start_char = int(line[1]), line[2], int(line[4])
            end_char = len(event_word) + start_char - 1
            data_dict["event_dict"][event_id] = {
                "mention": event_word,
                "start_char": start_char,
                "end_char": end_char,
            }
        elif type == "relation":
            event_id1, event_id2, rel_type = int(line[1]), int(line[2]), line[3]
            rel_id = get_relation_id(rel_type)
            data_dict["relation_dict"][(event_id1, event_id2)] = {}
            data_dict["relation_dict"][(event_id1, event_id2)]["relation"] = rel_id
        else:
            raise ValueError("File is not in HiEve tsvx format...")
    return data_dict


def hieve_file_reader(data_dir: Union[Path, str], file: str) -> Dict[str, Any]:
    data_dict = read_tsvx_file(data_dir, file)
    data_dict = document_to_sentences(data_dict) # sentence information update

    # Add sent_id as an attribute of event
    event_dict = data_dict["event_dict"]
    for event_id, event_dict_per_id in event_dict.items():
        start_char, end_char = event_dict_per_id["start_char"], event_dict_per_id["end_char"]
        sntc_id = sntc_id_lookup(data_dict, start_char, end_char)

        token_span_DOC = data_dict["sentences"][sntc_id]["token_span_DOC"]
        roberta_subword_span_DOC = data_dict["sentences"][sntc_id]["roberta_subword_span_DOC"]

        event_dict[event_id]["sent_id"] = sntc_id
        event_dict[event_id]["token_id"] = id_lookup(token_span_DOC, start_char)
        event_dict[event_id]["roberta_subword_id"] = id_lookup(roberta_subword_span_DOC, start_char)

    return data_dict


