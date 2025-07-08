
from sqlalchemy import create_engine
import pandas as pd
from config import DB_CONFIG

def load_training_data():
    db_url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
    
    engine = create_engine(db_url)
    query = """
    SELECT radius_mean, texture_mean, perimeter_mean, area_mean,
           smoothness_mean, compactness_mean, concavity_mean, diagnosis
    FROM data_training
    """
    df = pd.read_sql(query, engine)
    return df

