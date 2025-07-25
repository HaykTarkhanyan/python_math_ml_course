import pytest 
import pandas as pd
from elections import load_data, get_num_unique_regions, get_first_name

def test_load_data():
    df = load_data()
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) <= 1000, "DataFrame should contain at most 1000 rows"

@pytest.fixture # setup
def sample_df():
    df = load_data()
    return df

def test_num_unique_regions(sample_df):
    num_regions = get_num_unique_regions(sample_df)
    assert isinstance(num_regions, int), "Number of unique regions should be an integer"
    assert num_regions > 0, "There should be at least one unique region"

def test_first_name(sample_df):
    first_name = get_first_name(sample_df)
    assert isinstance(first_name, str), "First name should be a string"
    assert first_name != "", "First name should not be empty"
    
# teardown
@pytest.fixture
def sample_df_with_teardown(): # pytest -k "teardown"
    df = load_data()
    yield df
    # Teardown logic
    df.to_csv("teardown_elections.csv", index=False)

def test_first_name_teardown(sample_df_with_teardown):
    first_name = get_first_name(sample_df_with_teardown)
    assert isinstance(first_name, str), "First name should be a string"
    assert first_name != "", "First name should not be empty"
    
# ------------- Mocking -------------
# uv pip install pytest-mock

def test_num_unique_regions_mock(mocker):
    mock_df = pd.DataFrame({
        'marz': ['Region1', 'Region2', 'Region1', 'Region3', 'Region2']
    })
    # mocker.patch('elections.load_data', return_value=mock_df)
    mocker.patch('test_elections.load_data', return_value=mock_df)
    

    df = load_data()  # This will use the mocked return value
    print(df)
    num_regions = get_num_unique_regions(df)
    assert num_regions == 3, f"Mocked DataFrame should have 3 unique regions, but got {num_regions}"
