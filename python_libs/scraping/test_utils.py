import pytest
from utils import clean_name

def test_clean_name():
    input = "\n      Արթուր\n  Արտուշի\n  Սահակյան\n\n  "
    output = "Արթուր Արտուշի Սահակյան"
    
    assert clean_name(input) == output, f"Clean name failed, input {input}"
    