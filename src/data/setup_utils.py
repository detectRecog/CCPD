# standard library imports
import os

__all__ = ["PROVINCES", "ALPHABET", "ALPHABET_NUMBERS"]

os.environ["DATA_DIR"] = "/Users/timurcarstensen/PycharmProjects/CCPD/resources/data/"

if not os.getenv("DATA_DIR"):
    raise ValueError(
        "You must set the DATA_DIR environment variable to point to your data directory."
    )

PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
    "警",
    "学",
    "O",
]

ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "O",
]

ALPHABET_NUMBERS = ALPHABET + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]
