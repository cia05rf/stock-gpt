"""Model for sql database"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

from config import DB_PATH

#Start the engine and Session
engine = create_engine(
    f'sqlite:///{str(DB_PATH)}'
)
Session = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))

Base = declarative_base()